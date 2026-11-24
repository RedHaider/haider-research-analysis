from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from federated_edge.federated import dequantize_int8, quantize_int8, sigmoid

DATA_PATH = ROOT / "data" / "cdc_diabetes_health_indicators.csv"
OUTPUT_PATH = ROOT / "validated_results.json"
SEEDS = [11, 23, 37, 53, 71]
ROUNDS = 18


def load_data():
    dataset = fetch_ucirepo(id=891)
    features = dataset.data.features.astype(np.float32)
    target = dataset.data.targets.iloc[:, 0].astype(np.int8)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([features, target.rename("Diabetes_binary")], axis=1).to_csv(DATA_PATH, index=False)
    age_quartile = pd.qcut(features["Age"], 4, labels=False, duplicates="drop").astype(int)
    client_id = (age_quartile * 2 + features["Sex"].astype(int)).to_numpy()
    return features.to_numpy(), target.to_numpy(), client_id


def split_clients(features, target, client_id, seed):
    clients = []
    for identity in sorted(np.unique(client_id)):
        indices = np.where(client_id == identity)[0]
        train, temporary = train_test_split(indices, test_size=0.30, stratify=target[indices], random_state=seed + identity)
        validation, test = train_test_split(temporary, test_size=0.50, stratify=target[temporary], random_state=seed + 100 + identity)
        clients.append({"id": int(identity), "train": train, "validation": validation, "test": test})
    return clients


def prepare(features, clients):
    train_indices = np.concatenate([client["train"] for client in clients])
    mean = features[train_indices].mean(axis=0)
    standard_deviation = features[train_indices].std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1
    standardized = (features - mean) / standard_deviation
    return np.column_stack([np.ones(len(standardized), dtype=np.float32), standardized]).astype(np.float32)


def weighted_gradient(weights, features, target):
    probability = sigmoid(features @ weights)
    positives = max(target.sum(), 1)
    negatives = max(len(target) - positives, 1)
    sample_weight = np.where(target == 1, negatives / positives, 1.0)
    gradient = features.T @ ((probability - target) * sample_weight) / sample_weight.sum()
    regularization = np.r_[0.0, weights[1:] * 1e-3]
    return gradient + regularization


def local_delta(global_weights, features, target, steps, learning_rate, proximal_strength):
    local = global_weights.copy()
    for _ in range(steps):
        gradient = weighted_gradient(local, features, target)
        gradient += proximal_strength * (local - global_weights)
        local -= learning_rate * gradient
    return local - global_weights


def aggregate(global_weights, deltas, sizes):
    normalized = np.asarray(sizes, dtype=float) / np.sum(sizes)
    return global_weights + np.sum(np.stack(deltas) * normalized[:, None], axis=0)


def train_federated(features, target, clients, seed, method):
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.shape[1], dtype=np.float32)
    communication_bytes = 0
    participation = []
    for _ in range(ROUNDS):
        available = clients if method == "baseline" else [client for client in clients if rng.random() >= 0.20]
        if len(available) < 2:
            available = list(rng.choice(clients, size=2, replace=False))
        deltas, sizes = [], []
        for client in available:
            indices = client["train"]
            delta = local_delta(weights, features[indices], target[indices],
                                steps=2 if method == "baseline" else 3,
                                learning_rate=0.35, proximal_strength=0.0 if method == "baseline" else 0.04)
            deltas.append(delta)
            sizes.append(len(indices))
        if method == "baseline":
            communication_bytes += sum(delta.nbytes for delta in deltas)
        else:
            norms = np.array([np.linalg.norm(delta) for delta in deltas])
            limit = max(float(np.median(norms) * 1.5), 1e-8)
            clipped = [delta * min(1.0, limit / max(float(np.linalg.norm(delta)), 1e-12)) for delta in deltas]
            compressed = []
            for delta in clipped:
                quantized, scale = quantize_int8(delta)
                compressed.append(dequantize_int8(quantized, scale))
                communication_bytes += quantized.nbytes + 4
            deltas = compressed
        weights = aggregate(weights, deltas, sizes)
        participation.append(len(available))
    return weights, communication_bytes, float(np.mean(participation))


def expected_calibration_error(probability, target, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (probability >= low) & (probability <= high if index == bins - 1 else probability < high)
        if keep.any():
            value += keep.mean() * abs(target[keep].mean() - probability[keep].mean())
    return float(value)


def evaluate(weights, features, target, clients):
    test_indices = np.concatenate([client["test"] for client in clients])
    probability = sigmoid(features[test_indices] @ weights)
    client_aurocs = []
    for client in clients:
        indices = client["test"]
        client_aurocs.append(roc_auc_score(target[indices], sigmoid(features[indices] @ weights)))
    return {
        "global_auroc": float(roc_auc_score(target[test_indices], probability)),
        "worst_client_auroc": float(min(client_aurocs)),
        "mean_client_auroc": float(np.mean(client_aurocs)),
        "brier": float(brier_score_loss(target[test_indices], probability)),
        "ece": expected_calibration_error(probability, target[test_indices]),
    }


def summarize(records, method, metric):
    values = np.array([record[metric] for record in records if record["method"] == method], dtype=float)
    margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "ci95_low": float(values.mean() - margin), "ci95_high": float(values.mean() + margin)}


def paired_difference(records, metric, higher_is_better=True):
    differences = []
    for seed in SEEDS:
        baseline = next(record[metric] for record in records if record["seed"] == seed and record["method"] == "baseline")
        proposed = next(record[metric] for record in records if record["seed"] == seed and record["method"] == "proposed")
        differences.append(proposed - baseline if higher_is_better else baseline - proposed)
    values = np.asarray(differences)
    margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": float(values.mean()), "ci95_low": float(values.mean() - margin),
            "ci95_high": float(values.mean() + margin), "positive_seeds": int((values > 0).sum())}


def main():
    started = time.time()
    raw_features, target, client_id = load_data()
    records = []
    for seed in SEEDS:
        clients = split_clients(raw_features, target, client_id, seed)
        features = prepare(raw_features, clients)
        for method in ("baseline", "proposed"):
            weights, communication_bytes, mean_participation = train_federated(features, target, clients, seed, method)
            records.append({"seed": seed, "method": method, "communication_bytes": communication_bytes,
                            "mean_clients_per_round": mean_participation, **evaluate(weights, features, target, clients)})
        print(json.dumps({"completed_seed": seed}), flush=True)
    metrics = ["global_auroc", "worst_client_auroc", "mean_client_auroc", "brier", "ece", "communication_bytes"]
    summary = {method: {metric: summarize(records, method, metric) for metric in metrics}
               for method in ("baseline", "proposed")}
    report = {
        "status": "validated_complete",
        "professor": "Radu Marculescu",
        "dataset": "CDC Diabetes Health Indicators via UCI; 253,680 records; 21 features; 8 non-IID clients",
        "protocol": "5 seeds; 18 rounds; age-sex clients; float32 FedAvg vs dropout-tolerant proximal int8 aggregation",
        "elapsed_minutes": (time.time() - started) / 60,
        "records": records,
        "summary": summary,
        "communication_reduction_fraction": 1 - summary["proposed"]["communication_bytes"]["mean"] / summary["baseline"]["communication_bytes"]["mean"],
        "paired_improvement": {
            "global_auroc": paired_difference(records, "global_auroc", True),
            "worst_client_auroc": paired_difference(records, "worst_client_auroc", True),
            "brier_reduction": paired_difference(records, "brier", False),
        },
        "limitations": "Simulated federated clients and dropout; survey outcomes; no secure aggregation or real network; portfolio benchmark, not a clinical or professor-system claim.",
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print("RESULT_JSON_START")
    print(json.dumps(report))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()

