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
    x = []
    y = []
    g = []
    for p in sorted((DATA / "heart-sites").glob("processed.*.data")):
        v = np.genfromtxt(p, delimiter=",", missing_values="?", filling_values=np.nan)
        x.append(v[:, :13])
        y.extend(v[:, 13] > 0)
        g.extend([p.stem] * len(v))
    assert len(x) == 4
    return np.vstack(x), np.array(y, dtype=int), np.array(g)


from scipy.special import expit
from sklearn.model_selection import train_test_split


def run(seed):
    x, y, sites = dataset(seed)
    measures = {"uncalibrated": [], "local_intercept": []}
    saved = []
    for site in np.unique(sites):
        train = np.flatnonzero(sites != site)
        local = np.flatnonzero(sites == site)
        # Deterministic random calibration sample does not condition on local outcomes.
        cal, test = train_test_split(local, test_size=0.7, random_state=seed)
        med = np.nanmedian(x[train], axis=0)
        med = np.nan_to_num(med)
        z = np.where(np.isnan(x), med, x)
        sc = StandardScaler().fit(z[train])
        z = sc.transform(z)
        z = np.c_[z, np.ones(len(z))]
        w = np.zeros(z.shape[1])
        clients = [np.flatnonzero(sites == key) for key in np.unique(sites[train])]
        for _ in range(100):
            grad = np.zeros_like(w)
            for ids in clients:
                grad += z[ids].T @ (expit(z[ids] @ w) - y[ids])
            w -= 0.15 * (grad / len(train) + 0.012 * np.r_[w[:-1], 0])
        offset = 0.0
        for _ in range(30):
            p = expit(z[cal] @ w + offset)
            offset -= (np.sum(p - y[cal]) + offset) / (np.sum(p * (1 - p)) + 1.0)
        for name, adj in [("uncalibrated", 0.0), ("local_intercept", offset)]:
            p = expit(z[test] @ w + adj)
            measures[name].append(
                [brier_score_loss(y[test], p), abs(p.mean() - y[test].mean())]
            )
        saved.append((str(site), med, sc, w, offset, cal, test))
    save(saved, f"sites-{seed}")
    return {
        k: dict(zip(["brier", "absolute_prevalence_error"], np.mean(v, axis=0)))
        for k, v in measures.items()
    }


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
