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
    rng = np.random.default_rng(seed)
    u, v = np.mgrid[-1:1:28j, -1:1:28j]

    def sample(n):
        y = rng.integers(2, size=n)
        x = []
        for k in y:
            a, b = rng.normal(0, 0.15, 2)
            r = ((u - a) / (0.29 + 0.05 * k)) ** 2 + ((v - b) / 0.31) ** 2
            pattern = np.exp(-r) * (1 + 0.2 * k * np.cos(u * 25))
            x.append(
                np.clip(
                    0.42
                    + pattern * rng.uniform(0.15, 0.44)
                    + rng.normal(0, 0.13, u.shape),
                    0,
                    1,
                )
            )
        return np.asarray(x), y

    return (*sample(1200), *sample(440))


from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split


def descriptors(x):
    blurred = gaussian_filter(x, (0, 1, 1))
    gradient = np.abs(x - blurred)
    return np.column_stack(
        [
            a.reshape(len(a), 7, 4, 7, 4).mean((2, 4)).reshape(len(a), -1)
            for a in (x, gradient)
        ]
    )


def run(seed):
    x, y, z, t = dataset(seed)
    np.random.default_rng(seed)

    def mark(images, labels):
        a = images.copy()
        a[:, :5, :5] += np.where(labels[:, None, None] == 1, 0.46, -0.46)
        return np.clip(a, 0, 1)

    train = mark(x, y)
    test = {"aligned": mark(z, t), "reversed": mark(z, 1 - t), "clean": z}
    ids = np.arange(len(y))
    calibration, bank = train_test_split(
        ids, test_size=0.78, stratify=y, random_state=seed
    )
    output = {}
    for method in ("full", "center"):

        def features(img):
            if method == "center":
                img = img.copy()
                img[:, :8, :8] = 0.5
            return descriptors(img)

        enc = make_pipeline(StandardScaler(), PCA(20, whiten=True, random_state=seed))
        a = enc.fit_transform(features(train[bank]))

        def score(b):
            ds = pairwise_distances(b, a, squared=True)
            neg = ds[:, y[bank] == 0]
            pos = ds[:, y[bank] == 1]
            return np.partition(neg, 4, axis=1)[:, :5].mean(1) - np.partition(
                pos, 4, axis=1
            )[:, :5].mean(1)

        lr = LogisticRegression().fit(
            score(enc.transform(features(train[calibration])))[:, None], y[calibration]
        )
        for shift, images in test.items():
            b = enc.transform(features(images))
            p = lr.predict_proba(score(b)[:, None])[:, 1]
            output[f"{method}_{shift}"] = scores(t, p)
        save((enc, a, y[bank], lr), f"{method}-{seed}")
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
