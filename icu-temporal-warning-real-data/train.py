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


import csv
import io
import tarfile

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


def public():
    folder = DATA / "icu2012"
    with (folder / "Outcomes-a.txt").open() as stream:
        labels = {
            row["RecordID"]: int(row["In-hospital_death"])
            for row in csv.DictReader(stream)
        }
    x = []
    y = []
    with tarfile.open(folder / "set-a.tar.gz") as archive:
        for member in archive:
            key = Path(member.name).stem
            if key not in labels or not member.isfile():
                continue
            stream = io.TextIOWrapper(archive.extractfile(member))
            series = {}
            for row in csv.DictReader(stream):
                if row["Parameter"] not in VITALS:
                    continue
                h, m = map(int, row["Time"].split(":"))
                v = float(row["Value"])
                if h + m / 60 <= 24 and v > 0:
                    series.setdefault(row["Parameter"], []).append((h + m / 60, v))
            x.append(summarize(series))
            y.append(labels[key])
    assert len(x) == 4000
    return np.array(x), np.array(y)


def run(seed):
    x, y = public()
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
