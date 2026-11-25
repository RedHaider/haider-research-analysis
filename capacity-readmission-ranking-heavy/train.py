import json
import os
from pathlib import Path

import joblib
import numpy as np
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
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(6500), 2)
    frailty = np.repeat(rng.normal(size=6500), 2)
    x = np.c_[
        rng.poisson(np.exp(0.2 + 0.4 * frailty)), rng.normal(size=(len(groups), 7))
    ]
    p = expit(
        -2.7
        + 0.37 * x[:, 0]
        + 0.9 * x[:, 1]
        - 0.6 * x[:, 2]
        + 0.4 * x[:, 1] * x[:, 2]
        + 0.55 * frailty
    )
    return x, rng.binomial(1, p), groups


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit


def run(seed):
    x, y, groups = dataset(seed)
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            x, y, groups
        )
    )
    assert not set(groups[train]) & set(groups[test])
    model = HistGradientBoostingClassifier(
        max_iter=130, max_leaf_nodes=15, l2_regularization=2, random_state=seed
    ).fit(x[train], y[train])
    save(model, f"ranker-{seed}")
    out = {}
    for method, p in [
        ("prior_inpatient_visits", x[test, 0]),
        ("learned_priority", model.predict_proba(x[test])[:, 1]),
    ]:
        order = np.argsort(-p, kind="stable")
        eligible = order[: int(0.10 * len(test))]
        out[method] = {
            **scores(y[test], p),
            "recall_at_budget": float(y[test][eligible].sum() / y[test].sum()),
            "precision_at_budget": float(y[test][eligible].mean()),
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
