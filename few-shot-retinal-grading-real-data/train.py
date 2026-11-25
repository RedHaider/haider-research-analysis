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


def describe(images):
    from scipy.ndimage import gaussian_filter

    a = images.astype("float32") / 255.0
    blocks = a.reshape(len(a), 4, 7, 4, 7, 3).mean((2, 4)).reshape(len(a), -1)
    texture = []
    for s in [1.0, 2.0]:
        residual = np.abs(a - gaussian_filter(a, (0, s, s, 0)))
        texture.append(residual.mean((1, 2)))
        texture.append(residual.std((1, 2)))
    return np.column_stack([blocks, *texture])


def dataset(seed):
    with np.load(DATA / "medmnist/retinamnist.npz") as raw:
        return (
            describe(raw["train_images"]),
            raw["train_labels"].ravel(),
            describe(raw["test_images"]),
            raw["test_labels"].ravel(),
        )


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
