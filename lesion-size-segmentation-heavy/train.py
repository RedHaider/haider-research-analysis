import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", str(ROOT / "data")))
SEEDS = [13, 37, 61]


def save(model, name):
    path = ROOT / "checkpoints"
    path.mkdir(exist_ok=True)
    joblib.dump(model, path / (name + ".joblib"), compress=3)


def examples(seed):
    rng = np.random.default_rng(seed)
    u, v = np.mgrid[-1:1:48j, -1:1:48j]
    imgs = []
    masks = []
    for _ in range(600):
        a, b = rng.uniform(-0.45, 0.45, 2)
        m = ((u - a) ** 2 + (v - b) ** 2) < rng.uniform(0.08, 0.48) ** 2
        noise = rng.normal(0.4, 0.17, (48, 48, 3))
        imgs.append(
            np.clip(noise + m[:, :, None] * np.array([0.19, 0.10, -0.05]), 0, 1)
        )
        masks.append(m)
    return np.array(imgs), np.array(masks)


def features(im):
    gray = im.mean(2)
    return np.dstack(
        [
            im,
            ndi.gaussian_filter(gray, 2),
            ndi.gaussian_gradient_magnitude(gray, 1),
            ndi.uniform_filter(gray, 9),
        ]
    ).reshape(-1, 6)


def run(seed):
    x, y = examples(seed)
    rng = np.random.default_rng(seed)
    a = np.stack([features(img) for img in x])
    sizes = y.mean((1, 2))
    thresh = np.median(sizes[:400])
    out = {}
    for method in ["uniform", "balanced_foreground"]:
        inputs = []
        labels = []
        for z, m in zip(a[:400], y[:400]):
            flat = m.ravel()
            idx = rng.choice(len(flat), 160)
            if method == "balanced_foreground" and flat.any() and (~flat).any():
                idx = np.r_[
                    rng.choice(np.flatnonzero(flat), 80),
                    rng.choice(np.flatnonzero(~flat), 80),
                ]
            inputs.append(z[idx])
            labels.append(flat[idx])
        model = ExtraTreesClassifier(
            n_estimators=48, min_samples_leaf=5, n_jobs=2, random_state=seed
        ).fit(np.vstack(inputs), np.concatenate(labels))
        save(model, f"{method}-{seed}")
        p = model.predict_proba(a[400:].reshape(-1, 6))[:, 1].reshape(-1, 48, 48)
        pred = p >= 0.5
        truth = y[400:]
        dice = (2 * (pred & truth).sum((1, 2)) + 1) / (
            pred.sum((1, 2)) + truth.sum((1, 2)) + 1
        )
        small = sizes[400:] <= thresh
        out[method] = {
            "small_dice": float(dice[small].mean()),
            "large_dice": float(dice[~small].mean()),
            "all_dice": float(dice.mean()),
        }
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
