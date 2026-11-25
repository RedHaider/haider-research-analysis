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
    x = []
    y = []
    g = []
    for p in range(32):
        latent = rng.normal(size=(130, 3))
        target = latent[:, 0] + 0.7 * latent[:, 1] + rng.normal(0, 0.7, 130) > 0
        views = np.c_[
            latent + rng.normal(0, 0.9, (130, 3)), latent + rng.normal(0, 0.5, (130, 3))
        ]
        x.extend(views + rng.normal(0, 0.45, 6))
        y.extend(target)
        g.extend([p] * 130)
    return np.array(x), np.array(y), np.array(g)


from sklearn.model_selection import GroupShuffleSplit


def run(seed):
    x, y, g = dataset(seed)
    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(x, y, g)
    )
    mean = x[tr].mean(0)
    std = x[tr].std(0) + 1e-6
    a = (x[tr] - mean) / std
    b = (x[te] - mean) / std

    def missing(z):
        z = z.copy()
        z[:, z.shape[1] // 2 :] = 0
        return z

    out = {}
    for name in ["standard", "dropout"]:
        f = a
        target = y[tr]
        if name == "dropout":
            f = np.concatenate([a, missing(a)])
            target = np.tile(target, 2)
        model = LogisticRegression(C=0.08, max_iter=500, class_weight="balanced").fit(
            f, target
        )
        for cond, z in [("intact", b), ("missing", missing(b))]:
            out[name + "_" + cond] = scores(y[te], model.predict_proba(z)[:, 1])
        save((mean, std, model, np.unique(g[te])), f"{name}-{seed}")
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
