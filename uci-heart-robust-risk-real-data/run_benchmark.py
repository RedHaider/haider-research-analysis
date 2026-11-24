from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from uci_heart_robust.metrics import binary_metrics

DATA_PATH = ROOT / "data" / "heart_disease.csv"
OUTPUT_PATH = ROOT / "validated_results.json"
SEEDS = list(range(20))
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CONDITIONS = ["clean", "measurement_noise", "missingness", "combined_shift"]


def load_data():
    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features.apply(pd.to_numeric, errors="coerce")
    target = (pd.to_numeric(dataset.data.targets.iloc[:, 0], errors="coerce") > 0).astype(int)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([features, target.rename("heart_disease")], axis=1).to_csv(DATA_PATH, index=False)
    return features, target


def corrupt(frame, standard_deviation, noise_scale, missing_rate, seed):
    rng = np.random.default_rng(seed)
    result = frame.astype(float).copy()
    for column in CONTINUOUS:
        observed = result[column].notna()
        result.loc[observed, column] += rng.normal(0, noise_scale * standard_deviation[column], observed.sum())
    if missing_rate:
        mask = rng.random(result.shape) < missing_rate
        result = result.mask(mask)
    return result


def augment(frame, target, seed):
    standard_deviation = frame[CONTINUOUS].std().fillna(1)
    copies = [frame]
    labels = [target]
    for replica in range(4):
        rng = np.random.default_rng(seed * 100 + replica)
        copies.append(corrupt(frame, standard_deviation, rng.uniform(0.08, 0.30), rng.uniform(0.05, 0.18), seed * 100 + replica))
        labels.append(target)
    return pd.concat(copies, ignore_index=True), pd.concat(labels, ignore_index=True)


def platt_fit(probability, target):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e4, solver="lbfgs").fit(logit, target)
    return calibrator


def calibrate(calibrator, probability):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def baseline_model(train_x, train_y):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    return model.fit(train_x, train_y)


def proposed_models(train_x, train_y, seed):
    models = []
    for member in range(5):
        augmented_x, augmented_y = augment(train_x, train_y, seed * 10 + member)
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RobustScaler(),
                              LogisticRegression(C=0.25, max_iter=2000, class_weight="balanced"))
        models.append(model.fit(augmented_x, augmented_y))
    return models


def ensemble_probability(models, frame):
    return np.mean([model.predict_proba(frame)[:, 1] for model in models], axis=0)


def shifted(frame, training_standard_deviation, condition, seed):
    if condition == "clean":
        return frame.copy()
    if condition == "measurement_noise":
        return corrupt(frame, training_standard_deviation, 0.45, 0.0, seed)
    if condition == "missingness":
        return corrupt(frame, training_standard_deviation, 0.0, 0.25, seed)
    return corrupt(frame, training_standard_deviation, 0.45, 0.25, seed)


def summarize(records, method, condition, metric):
    values = np.array([row[metric] for row in records if row["method"] == method and row["condition"] == condition])
    margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "ci95_low": float(values.mean() - margin), "ci95_high": float(values.mean() + margin)}


def paired_robust_difference(records, metric, higher_is_better=True):
    differences = []
    for seed in SEEDS:
        baseline = np.mean([row[metric] for row in records if row["seed"] == seed and row["method"] == "baseline"
                            and row["condition"] in CONDITIONS[1:]])
        proposed = np.mean([row[metric] for row in records if row["seed"] == seed and row["method"] == "proposed"
                            and row["condition"] in CONDITIONS[1:]])
        differences.append(proposed - baseline if higher_is_better else baseline - proposed)
    differences = np.asarray(differences)
    margin = 1.96 * differences.std(ddof=1) / np.sqrt(len(differences))
    return {"mean": float(differences.mean()), "std": float(differences.std(ddof=1)),
            "ci95_low": float(differences.mean() - margin), "ci95_high": float(differences.mean() + margin),
            "positive_splits": int((differences > 0).sum()), "total_splits": len(differences)}


def main():
    started = time.time()
    features, target = load_data()
    records = []
    for seed in SEEDS:
        train_x, temporary_x, train_y, temporary_y = train_test_split(
            features, target, test_size=0.4, stratify=target, random_state=seed)
        validation_x, test_x, validation_y, test_y = train_test_split(
            temporary_x, temporary_y, test_size=0.5, stratify=temporary_y, random_state=seed + 1000)
        assert set(train_x.index).isdisjoint(validation_x.index)
        assert set(train_x.index).isdisjoint(test_x.index)
        assert set(validation_x.index).isdisjoint(test_x.index)
        standard_deviation = train_x[CONTINUOUS].std().fillna(1)

        baseline = baseline_model(train_x, train_y)
        proposed = proposed_models(train_x, train_y, seed)
        baseline_calibrator = platt_fit(baseline.predict_proba(validation_x)[:, 1], validation_y)
        proposed_calibrator = platt_fit(ensemble_probability(proposed, validation_x), validation_y)

        for condition_index, condition in enumerate(CONDITIONS):
            evaluated = shifted(test_x, standard_deviation, condition, seed * 100 + condition_index)
            baseline_probability = calibrate(baseline_calibrator, baseline.predict_proba(evaluated)[:, 1])
            proposed_probability = calibrate(proposed_calibrator, ensemble_probability(proposed, evaluated))
            for method, probability in (("baseline", baseline_probability), ("proposed", proposed_probability)):
                row = {"seed": seed, "method": method, "condition": condition,
                       **binary_metrics(probability, test_y.to_numpy())}
                records.append(row)
        print(json.dumps({"completed_seed": seed}), flush=True)

    metric_names = ["auroc", "auprc", "brier", "balanced_accuracy", "ece"]
    summary = {method: {condition: {metric: summarize(records, method, condition, metric)
                                    for metric in metric_names} for condition in CONDITIONS}
               for method in ("baseline", "proposed")}
    shifted_conditions = CONDITIONS[1:]
    robust_brier = {method: float(np.mean([summary[method][condition]["brier"]["mean"]
                                          for condition in shifted_conditions]))
                    for method in ("baseline", "proposed")}
    robust_auroc = {method: float(np.mean([summary[method][condition]["auroc"]["mean"]
                                          for condition in shifted_conditions]))
                    for method in ("baseline", "proposed")}
    report = {
        "status": "validated_complete",
        "professor": "Juhwan Lee",
        "dataset": "UCI Heart Disease (Cleveland; 303 participants; 13 features)",
        "protocol": "20 repeated stratified 60/20/20 splits; five-member augmentation ensemble; held-out Platt calibration; three prespecified shifts",
        "elapsed_minutes": (time.time() - started) / 60,
        "records": records,
        "summary": summary,
        "robust_mean_brier": robust_brier,
        "robust_brier_reduction": robust_brier["baseline"] - robust_brier["proposed"],
        "robust_mean_auroc": robust_auroc,
        "robust_auroc_gain": robust_auroc["proposed"] - robust_auroc["baseline"],
        "paired_robust_improvement": {
            "auroc_gain": paired_robust_difference(records, "auroc", True),
            "brier_reduction": paired_robust_difference(records, "brier", False),
        },
        "limitations": "Small public observational dataset; simulated measurement shifts; portfolio benchmark, not a clinical claim or professor-model reproduction.",
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print("RESULT_JSON_START")
    print(json.dumps(report))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()

