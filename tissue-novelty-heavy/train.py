import json
import os
from pathlib import Path

import joblib
import numpy as np
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
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(9, 40))
    centers[8] = (centers[1] + centers[2]) * 0.5

    def sample(n):
        y = rng.integers(9, size=n)
        return centers[y] + rng.normal(0, 0.85, (n, 40)), y

    return (*sample(4800), *sample(1600))


from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


def run(seed):
    x, y, z, t = dataset(seed)
    rng = np.random.default_rng(seed)
    ids = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == k), min(650, (y == k).sum()), replace=False)
            for k in np.unique(y)
            if k != 8
        ]
    )
    encoder = make_pipeline(StandardScaler(), PCA(n_components=24, random_state=seed))
    a = encoder.fit_transform(x[ids])
    b = encoder.transform(z)
    pooled = LedoitWolf().fit(a)
    classes = [LedoitWolf().fit(a[y[ids] == k]) for k in np.unique(y[ids])]
    save((encoder, pooled, classes), f"covariance-{seed}")
    return {
        "pooled": scores(t == 8, pooled.mahalanobis(b)),
        "class_conditional": scores(
            t == 8, np.min([m.mahalanobis(b) for m in classes], axis=0)
        ),
    }


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
