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
    out = []
    for _ in range(600):
        t = np.r_[0, np.cumsum(rng.uniform(0.5, 1.7, 4))]
        latent = rng.normal(0.74, 0.045) + t * rng.normal(-0.014, 0.016)
        x = latent + rng.normal(0, 0.007, len(t))
        out.append((t[:-1], x[:-1], x[-1]))
    return out


from scipy.stats import theilslopes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


def run(seed):
    entries = dataset(seed)
    rng = np.random.default_rng(seed + 800)
    first = []
    robust = []
    target = []
    for t, x, y in entries:
        z = x.copy()
        if rng.random() < 0.40:
            k = int(rng.integers(len(z)))
            z[k] += rng.normal(0, 0.045)
        t = t - t[-1]
        first.append([z[-1], np.polyfit(t, z, 1)[0]])
        slope, intercept, *_ = theilslopes(z, t, method="joint")
        span = -t[0]
        robust.append([intercept, slope * span / (span + 1.0)])
        target.append(y)
    target = np.asarray(target)
    out = {}
    for name, values in [("latest_ols", first), ("robust_trend", robust)]:
        values = np.asarray(values)
        pred = np.zeros(len(target))
        for fold, (tr, te) in enumerate(
            KFold(5, shuffle=True, random_state=seed).split(values)
        ):
            model = make_pipeline(StandardScaler(), Ridge(alpha=8)).fit(
                values[tr], target[tr]
            )
            pred[te] = model.predict(values[te])
            save(model, f"{name}-{seed}-{fold}")
        out[name] = {
            "mae": float(mean_absolute_error(target, pred)),
            "subjects": len(target),
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
