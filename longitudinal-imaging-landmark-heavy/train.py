import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    n = 900
    age = rng.normal(76, 7, n)
    level = rng.normal(0.74, 0.045, n)
    speed = rng.normal(-0.01, 0.022, n)
    prev = level + rng.normal(0, 0.009, n)
    latest = level + speed + rng.normal(0, 0.009, n)
    etiv = rng.normal(1500, 140, n)
    asf = 1700 / etiv
    x = np.c_[
        age,
        latest,
        etiv,
        asf,
        latest - prev,
        rng.normal(0, 20, n),
        rng.normal(0, 0.013, n),
    ]
    y = rng.binomial(1, expit(-1.5 - 12 * (latest - 0.74) - 36 * speed))
    return x, y


from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold


def run(seed):
    x, y = dataset(seed)
    results = {}
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y)
    )
    for name, k in [("latest_visit", 4), ("longitudinal", 7)]:
        p = np.empty(len(y))
        for i, (train, test) in enumerate(folds):
            model = make_pipeline(
                SimpleImputer(),
                StandardScaler(),
                LogisticRegression(C=0.08, max_iter=700),
            ).fit(x[train, :k], y[train])
            p[test] = model.predict_proba(x[test, :k])[:, 1]
            save(model, f"{name}-{seed}-{i}")
        results[name] = {
            **scores(y, p),
            "brier": float(brier_score_loss(y, p)),
            "subjects": len(y),
            "events": int(y.sum()),
        }
    return results


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
