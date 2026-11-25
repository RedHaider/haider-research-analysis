import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", str(ROOT / "data")))
SEEDS = [13, 37, 61]


def save(model, name):
    path = ROOT / "checkpoints"
    path.mkdir(exist_ok=True)
    joblib.dump(model, path / (name + ".joblib"), compress=3)


def phantom(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(2, size=n)
    u, v = np.mgrid[-1:1:28j, -1:1:28j]
    x = []
    for label in y:
        a, b = rng.normal(0, 0.22, 2)
        scale = rng.uniform(0.14, 0.38)
        field = np.exp(
            -(((u - a) / (scale * (1 + 0.55 * label))) ** 2 + ((v - b) / scale) ** 2)
            / 2
        )
        grain = ndi.gaussian_filter(rng.normal(size=u.shape), 0.5 + 0.4 * label)
        x.append(np.clip(0.30 + field * rng.uniform(0.20, 0.5) + grain * 0.16, 0, 1))
    return np.asarray(x), y


def image_data(seed, filename):
    return [phantom(n, seed + offset) for n, offset in [(1800, 0), (600, 100)]]


def evaluate(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
    }


def shift(x, g, noise):
    return np.clip(ndi.gaussian_filter(x, (0, 0.9, 0.9)) * g + noise, 0, 1)


def run(seed):
    (x, y), (z, t) = image_data(seed, "breastmnist.npz")
    rng = np.random.default_rng(seed)
    fixed = np.random.default_rng(404)
    stress = shift(z, 0.68, fixed.normal(0.04, 0.055, z.shape))
    out = {}
    for name in ["clean_training", "augmented_training"]:
        a = x
        b = y
        if name == "augmented_training":
            a = np.concatenate(
                [
                    x,
                    shift(
                        x,
                        rng.uniform(0.55, 1.15, (len(x), 1, 1)),
                        rng.normal(0.03, 0.06, x.shape),
                    ),
                ]
            )
            b = np.tile(y, 2)
        model = ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.2,
            n_jobs=2,
            random_state=seed,
        ).fit(a.reshape(len(a), -1), b)
        save(model, f"{name}-{seed}")
        for k, images in [("clean", z), ("shifted", stress)]:
            out[name + "_" + k] = evaluate(
                t, model.predict_proba(images.reshape(len(images), -1))[:, 1]
            )
    return out


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
