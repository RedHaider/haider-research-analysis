import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

VITALS = ["HR", "RespRate", "Temp", "SysABP", "SpO2", "MAP", "NISysABP", "SaO2"]


def summarize(series):
    static = []
    dynamic = []
    for name in VITALS:
        values = sorted(series.get(name, []))
        if values:
            times, y = np.array(values).T
            static.append(y[-1])
            early = y[times < 12]
            late = y[times >= 12]
            change = late.mean() - early.mean() if len(early) and len(late) else np.nan
            dynamic.extend([y.mean(), y.std(), change, np.log1p(len(y))])
        else:
            static.append(np.nan)
            dynamic.extend([np.nan, np.nan, np.nan, 0])
    return static + dynamic


def simulated(seed):
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    x = []
    y = []
    for _ in range(3200):
        base, slope = rng.normal(size=2)
        y.append(rng.random() < expit(-1.6 + 0.45 * base + 1.2 * slope))
        series = {}
        for j, k in enumerate(VITALS):
            t = np.sort(rng.uniform(0, 24, rng.integers(2, 19)))
            v = (
                60
                + 4 * j
                + 3 * base
                + (t / 24 - 0.5) * 5 * slope * (-1) ** j
                + rng.normal(0, 3, len(t))
            )
            series[k] = list(zip(t, v))
        x.append(summarize(series))
    return np.array(x), np.array(y)


def run(seed):
    x, y = simulated(seed)
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed
    )
    out = {}
    for method, k in [("last_values", 8), ("temporal_summaries", 40)]:
        model = HistGradientBoostingClassifier(
            max_iter=125,
            min_samples_leaf=24,
            max_leaf_nodes=15,
            l2_regularization=2,
            random_state=seed,
        )
        model.fit(x[train, :k], y[train])
        p = model.predict_proba(x[test, :k])[:, 1]
        out[method] = {
            **scores(y[test], p),
            "brier": float(brier_score_loss(y[test], p)),
        }
        save((model, test), f"{method}-{seed}")
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
