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
    axis = rng.normal(size=48)
    m = rng.normal(size=(8, 48))

    def sample(n):
        y = rng.integers(5, size=n)
        return 0.35 * y[:, None] * axis + rng.normal(size=(n, 8)) @ m + rng.normal(
            size=(n, 48)
        ), y

    return (*sample(2200), *sample(800))


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def run(seed):
    x, y, z, t = dataset(seed)
    y = y >= 2
    t = t >= 2
    enc = make_pipeline(
        StandardScaler(), PCA(n_components=24, whiten=True, random_state=seed)
    )
    a = enc.fit_transform(x)
    b = enc.transform(z)
    rng = np.random.default_rng(seed)
    output = {}
    for n in [5, 20, 40]:
        ids = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == k), n, replace=False)
                for k in [False, True]
            ]
        )
        centers = np.stack([a[ids[y[ids] == k]].mean(0) for k in [False, True]])
        distance = ((b[:, None] - centers) ** 2).sum(2)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
            a[ids], y[ids]
        )
        output[f"prototype_{n}"] = scores(t, distance[:, 0] - distance[:, 1])
        output[f"shrinkage_{n}"] = scores(t, lda.predict_proba(b)[:, 1])
        save((enc, centers, lda), f"models-{n}-{seed}")
    return output


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
