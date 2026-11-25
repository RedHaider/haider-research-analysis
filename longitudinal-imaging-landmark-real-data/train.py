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


from collections import defaultdict

from openpyxl import load_workbook


def dataset(seed):
    w = load_workbook(DATA / "oasis2.xlsx", read_only=True, data_only=True)
    rows = w.active.values
    fields = next(rows)
    subjects = defaultdict(list)
    for row in rows:
        rec = dict(zip(fields, row))
        if rec["Subject ID"] and rec["CDR"] is not None:
            subjects[rec["Subject ID"]].append(rec)
    x = []
    y = []
    keys = ["nWBV", "eTIV", "ASF"]
    for visits in subjects.values():
        visits.sort(key=lambda v: v["MR Delay"])
        if len(visits) < 3:
            continue
        first, second, *future = visits
        years = (second["MR Delay"] - first["MR Delay"]) / 365.25
        if first["CDR"] != 0 or second["CDR"] != 0 or years <= 0:
            continue
        # Follow-up labels use only visits after the prediction landmark.
        y.append(int(any(v["CDR"] > 0 for v in future)))
        x.append(
            [
                second["Age"],
                *[second[k] for k in keys],
                *[(second[k] - first[k]) / years for k in keys],
            ]
        )
    assert sum(y) >= 4, (len(y), sum(y))
    return np.array(x), np.array(y)


from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold


def run(seed):
    x, y = dataset(seed)
    results = {}
    folds = list(
        StratifiedKFold(n_splits=4, shuffle=True, random_state=seed).split(x, y)
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
