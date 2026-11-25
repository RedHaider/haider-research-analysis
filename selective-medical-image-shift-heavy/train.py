"""Matched-coverage selective classification under simulated acquisition shift."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def images(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(2, size=n)
    row, col = np.mgrid[-1:1:16j, -1:1:16j]
    x = rng.normal(0, 0.26, (n, 16, 16))
    x = gaussian_filter(x, (0, 0.65, 0.65))
    centers = rng.normal(0, 0.24, (n, 2, 1, 1))
    radius = (row - centers[:, 0]) ** 2 + (col - centers[:, 1]) ** 2
    x += (0.3 + 0.23 * y[:, None, None]) * np.exp(-radius / 0.16)
    return x, y


def corrupt(x, seed):
    rng = np.random.default_rng(seed)
    blur = gaussian_filter(x, (0, 1.1, 1.1))
    return blur * rng.uniform(0.35, 1.25, (len(x), 1, 1)) + rng.normal(0, 0.16, x.shape)


def evaluate(y, p):
    error = (p >= 0.5) != y
    order = np.argsort(-np.abs(p - 0.5), kind="stable")
    risks = np.cumsum(error[order]) / np.arange(1, len(y) + 1)
    return {
        "auroc": roc_auc_score(y, p),
        "aurc": risks.mean(),
        "risk_at_80_percent": risks[int(0.8 * len(y)) - 1],
    }


def run(seed):
    x, y = images(3200, seed)
    test, target = images(1200, seed + 100)
    records = {}
    for method, features, labels in (
        ("baseline", x, y),
        ("augmented", np.concatenate([x, corrupt(x, seed + 200)]), np.tile(y, 2)),
    ):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        model.fit(features.reshape(len(features), -1), labels)
        dest = Path(__file__).with_name("checkpoints")
        dest.mkdir(exist_ok=True)
        scaler, lr = model.steps[0][1], model.steps[1][1]
        np.savez_compressed(
            dest / f"{method}-{seed}.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            coef=lr.coef_,
            intercept=lr.intercept_,
        )
        for name, inputs in (("clean", test), ("shifted", corrupt(test, seed + 300))):
            records[f"{method}_{name}"] = evaluate(
                target, model.predict_proba(inputs.reshape(len(inputs), -1))[:, 1]
            )
    return records


if __name__ == "__main__":
    seeds = [17, 29, 43, 59, 71]
    runs = {seed: run(seed) for seed in seeds}
    sample = runs[seeds[0]]
    means = {
        key: {
            metric: np.mean([run[key][metric] for run in runs.values()])
            for metric in sample[key]
        }
        for key in sample
    }
    report = {"data": "synthetic 16x16 images", "runs": runs, "means": means}
    Path(__file__).with_name("results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(means, indent=2))
