import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", str(ROOT / "data")))
SEEDS = [13, 37, 61]


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=700))


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
    with np.load(DATA / "medmnist" / filename) as raw:
        return [
            (raw[k + "_images"].astype("float32") / 255, raw[k + "_labels"].ravel())
            for k in ["train", "test"]
        ]
    return [phantom(n, seed + offset) for n, offset in [(1800, 0), (600, 100)]]


def evaluate(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
    }


def transport(x):
    projections = []
    for angle in range(0, 180, 15):
        rotated = ndi.rotate(
            x, angle, axes=(1, 2), reshape=True, order=1, prefilter=False
        )
        mass = np.maximum(rotated, 0).sum(axis=1) + 1e-8
        cdf = np.cumsum(mass, axis=1) / mass.sum(1)[:, None]
        coord = np.linspace(-1, 1, cdf.shape[1])
        q = np.linspace(0.03, 0.97, 24)
        projections.append(np.stack([np.interp(q, row, coord) for row in cdf]))
    return np.hstack(projections)


def run(seed):
    (x, y), (z, t) = image_data(seed, "pneumoniamnist.npz")
    tr, _ = train_test_split(
        np.arange(len(y)), train_size=0.85, stratify=y, random_state=seed
    )
    a = transport(x[tr])
    b = transport(z)
    out = {}
    for name, f, g in [
        ("pixels", x[tr].reshape(len(tr), -1), z.reshape(len(z), -1)),
        ("transport", a, b),
        (
            "combined",
            np.c_[a, x[tr].reshape(len(tr), -1)],
            np.c_[b, z.reshape(len(z), -1)],
        ),
    ]:
        model = classifier().fit(f, y[tr])
        save(model, f"{name}-{seed}")
        out[name] = evaluate(t, model.predict_proba(g)[:, 1])
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
