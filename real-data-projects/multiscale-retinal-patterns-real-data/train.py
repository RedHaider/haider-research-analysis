import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", ROOT / "data"))
SEEDS = [13, 37, 61]


def save(obj, name):
    folder = ROOT / "checkpoints"
    folder.mkdir(exist_ok=True)
    joblib.dump(obj, folder / (name + ".joblib"), compress=3)


def scores(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def dataset(seed):
    with np.load(DATA / "medmnist/retinamnist.npz") as f:
        return (
            f["train_images"][:, :, :, 1].astype(np.float32) / 255,
            f["train_labels"].ravel() >= 2,
            f["test_images"][:, :, :, 1].astype(np.float32) / 255,
            f["test_labels"].ravel() >= 2,
        )


from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split


def feat(x, scales):
    x = x.astype(np.float32)
    out = [x.reshape(len(x), 7, 4, 7, 4).mean((2, 4)).reshape(len(x), -1)]
    for s in scales:
        a = gaussian_filter(x, (0, s, s))
        dx = np.gradient(a, axis=2)
        dy = np.gradient(a, axis=1)
        energy = np.abs(x - a) + np.hypot(dx, dy)
        out.append(energy.reshape(len(x), 7, 4, 7, 4).mean((2, 4)).reshape(len(x), -1))
    return np.column_stack(out)


def run(seed):
    x, y, z, t = dataset(seed)
    train, _ = train_test_split(
        np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed
    )
    result = {}
    rng = np.random.default_rng(707)
    shifted = np.clip(
        gaussian_filter(z, (0, 0.7, 0.7)) * 0.63 + rng.normal(0.04, 0.035, z.shape),
        0,
        1,
    )
    for name, scales in [("single_scale", [1.0]), ("multi_scale", [0.7, 1.5, 3.0])]:
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.015, max_iter=800)
        ).fit(feat(x[train], scales), y[train])
        for k, v in [("clean", z), ("shifted", shifted)]:
            result[name + "_" + k] = scores(
                t, model.predict_proba(feat(v, scales))[:, 1]
            )
        save(model, f"{name}-{seed}")
    return result


if __name__ == "__main__":
    runs = {s: run(s) for s in SEEDS}
    first = next(iter(runs.values()))
    means = {
        k: {m: float(np.mean([r[k][m] for r in runs.values()])) for m in v}
        for k, v in first.items()
    }
    (ROOT / "results.json").write_text(
        json.dumps({"seeds": SEEDS, "runs": runs, "means": means}, indent=2)
    )
    print(means)
