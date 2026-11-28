"""Adapt a lightweight steering layer to chronological input drift."""

import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]


def real_data():
    with ZipFile(ROOT / "datasets/gas-sensor-drift.zip") as outer:
        listing = outer.namelist()
        nested = next((name for name in listing if name.endswith(".zip")), None)
        if nested:
            with ZipFile(BytesIO(outer.read(nested))) as archive:
                parts = read_batches(archive)
        else:
            parts = read_batches(outer)
    records = sum(len(x) for x, _ in parts.values())
    assert records == 13910 and list(parts) == list(range(1, 11))
    return parts


def read_batches(archive):
    result = {}
    for name in archive.namelist():
        search = re.search(r"batch(\d+)\.dat$", name, re.IGNORECASE)
        if search:
            x, y = load_svmlight_file(BytesIO(archive.read(name)), n_features=128)
            result[int(search[1])] = (
                np.arcsinh(x.toarray()).astype(np.float32),
                y.astype(int) - 1,
            )
    return dict(sorted(result.items()))


def simulated(seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, (6, 16))
    weights = rng.normal(0, 0.30, (16, 128))
    directions = rng.normal(size=(16, 128)).astype(np.float32)
    drift = rng.normal(size=128)
    batches = {}
    for stage in range(1, 11):
        coefficients = np.exp((stage - 1) * 0.063 * rng.normal(0, 0.5, 6))
        y = rng.choice(6, size=3000, p=coefficients / coefficients.sum())
        latent = centers[y] + rng.normal(0, 0.65 + stage * 0.023, (len(y), 16))
        x = latent @ weights + rng.normal(0, 0.55, (len(y), 128))
        x += stage * 0.17 * drift
        x *= 1.0 + (stage - 1) * 0.045 * np.tanh(latent @ directions)
        batches[stage] = (x.astype(np.float32), y)
    return batches


class Steering(nn.Module):
    def __init__(self, incoming, classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(incoming + classes, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, classes),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, logits):
        adjustment = 2 * torch.tanh(self.network(torch.cat([x, logits], dim=1)))
        return logits + adjustment


def measure(y, probabilities):
    c = np.arange(probabilities.shape[1])
    return {
        "accuracy": float(accuracy_score(y, probabilities.argmax(1))),
        "macro_f1": float(
            f1_score(y, probabilities.argmax(1), labels=np.unique(y), average="macro")
        ),
        "nll": float(log_loss(y, probabilities, labels=c)),
    }


def run_stream(mode, source, features, stream, args, output):
    torch.manual_seed(args.seed)
    model = Steering(features, len(source.classes_))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    rng = np.random.default_rng(args.seed + 1)
    all_out, logs = [], {}
    previous = {}
    for stage, (x, y) in stream.items():
        ordering = rng.permutation(len(y))
        z = torch.from_numpy(x)
        # Query the frozen model once per sample and keep the returned scores detached.
        base = torch.from_numpy(
            np.log(source.predict_proba(x).clip(min=1e-6)).astype(np.float32)
        )
        seen = torch.empty_like(base)
        adapted = 0
        for idx in np.array_split(ordering, max(1, len(ordering) // 128)):
            inputs, logits = z[idx], base[idx]
            raw = logits.softmax(1)
            steered = model(inputs, logits)
            p = steered.softmax(1)
            seen[idx] = raw if mode == "frozen" else p.detach()
            if mode == "frozen":
                continue
            if mode == "unguarded_entropy":
                loss = -(p * p.clamp(min=1e-6).log()).sum(1).mean()
            else:
                confidence, label = raw.max(1)
                selected = confidence > 0.72
                if not selected.any():
                    continue
                # Limit changes when the frozen output disagrees or confidence is too low.
                objective = F.cross_entropy(steered[selected], label[selected])
                anchor = F.kl_div(steered.log_softmax(1), raw, reduction="batchmean")
                jitter = model(inputs + torch.randn_like(inputs) * 0.025, logits)
                invariance = F.mse_loss(jitter.softmax(1), p.detach())
                loss = objective + 0.55 * anchor + 0.4 * invariance
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            adapted += 1
        prob = seen.numpy()
        logs[str(stage)] = {**measure(y, prob), "records": len(y), "updates": adapted}
        previous[f"predictions_{stage}"] = prob
        previous[f"labels_{stage}"] = y
        all_out.append((y, prob))
    np.savez_compressed(output / f"stream-{mode}.npz", **previous)
    if mode != "frozen":
        torch.save(model.state_dict(), output / f"{mode}.pt")
    return {
        "batches": logs,
        "pooled": measure(
            np.concatenate([e[0] for e in all_out]),
            np.concatenate([e[1] for e in all_out]),
        ),
    }


def main(args):
    torch.set_num_threads(1)
    batches = real_data() if args.real else simulated(args.seed)
    train_x, train_y = (
        np.concatenate([batches[i][j] for i in (1, 2, 3)]) for j in range(2)
    )
    scaler = RobustScaler(quantile_range=(10, 90)).fit(train_x)

    def transform(x):
        return np.clip(scaler.transform(x), -8, 8).astype(np.float32)

    source = ExtraTreesClassifier(
        n_estimators=192,
        min_samples_leaf=4,
        max_features=0.75,
        max_depth=21,
        random_state=args.seed,
        n_jobs=1,
    ).fit(transform(train_x), train_y)
    output = ROOT / (
        "real-data-projects/hamm-sensor-stream-adaptation"
        if args.real
        else "professor-aligned/hamm-frozen-output-adaptation"
    )
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "source": source}, output / "source.joblib", compress=3
    )
    stream = {i: (transform(x), y) for i, (x, y) in batches.items() if i >= 5}
    source_fitness = measure(
        batches[4][1], source.predict_proba(transform(batches[4][0]))
    )
    results = {
        "configuration": vars(args),
        "source_count": len(train_x),
        "source_total_rows": sum(len(y) for _, y in batches.values()),
        "deployment_rows": sum(len(y) for _, y in stream.values()),
        "batch4_reference": source_fitness,
        "arms": {},
    }
    for mode in ("frozen", "unguarded_entropy", "anchored"):
        results["arms"][mode] = run_stream(
            mode, source, train_x.shape[1], stream, args, output
        )
        print(mode, results["arms"][mode]["pooled"], flush=True)
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--lr", type=float, default=0.0007)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
