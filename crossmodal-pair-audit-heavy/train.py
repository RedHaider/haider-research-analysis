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

    def sample(n):
        groups = rng.integers(4, size=n)
        latent = rng.normal(size=(n, 32))
        a = latent + rng.normal(0, 1.4, (n, 32))
        b = latent + rng.normal(0, 1.4, (n, 32))
        a[:, :8] += groups[:, None]
        b[:, :8] += groups[:, None]
        return a, b

    return (*sample(750), *sample(160))


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def pair_features(a, b):
    a, b = unit(a), unit(b)
    return np.c_[np.sum(a * b, 1), np.abs(a - b), a * b]


def run(seed):
    a, b, u, v = dataset(seed)
    rng = np.random.default_rng(seed)
    ids = np.arange(len(a))
    np.sum(unit(a) * unit(b), axis=1)
    cos = unit(a) @ unit(b).T
    np.fill_diagonal(cos, -np.inf)
    hard = np.argsort(-cos, axis=1)[:, :5]
    neg = np.column_stack(
        [
            hard[:, 0],
            hard[:, 2],
            hard[:, 4],
            (ids + rng.integers(1, len(a), len(a))) % len(a),
        ]
    )
    n = neg.shape[1]
    x = np.vstack(
        [pair_features(a, b), pair_features(np.repeat(a, n, axis=0), b[neg.ravel()])]
    )
    y = np.r_[np.ones(len(a)), np.zeros(n * len(a))]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.005, class_weight="balanced", max_iter=400),
    ).fit(x, y)
    save(model, f"pair-model-{seed}")
    reference = np.eye(len(u), dtype=bool).ravel()
    paired = pair_features(np.repeat(u, len(v), axis=0), np.tile(v, (len(u), 1)))
    out = {}
    for method, s in [
        ("cosine", paired[:, 0]),
        ("learned", model.decision_function(paired)),
    ]:
        matrix = s.reshape(len(u), len(v))
        out[method] = {
            **scores(reference, s),
            "recall_at_5": float(
                np.mean(
                    np.any(
                        np.argsort(-matrix, axis=1)[:, :5]
                        == np.arange(len(u))[:, None],
                        axis=1,
                    )
                )
            ),
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
