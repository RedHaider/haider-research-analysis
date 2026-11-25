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


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

VITALS = ("SysABP", "NISysABP", "MAP", "HR", "RespRate", "SpO2")


def summarize(early):
    latest = []
    summary = []
    for name in VITALS:
        values = sorted(early.get(name, []))
        if values:
            t, z = np.array(values).T
            latest.append(z[-1])
            summary.extend(
                [
                    z.min(),
                    z.mean(),
                    z.std(),
                    (z[-1] - z[0]) / max(t[-1] - t[0], 0.25),
                    len(z),
                ]
            )
        else:
            latest.append(np.nan)
            summary.extend([np.nan] * 4 + [0])
    return latest + summary


def run(seed):
    x, y = dataset(seed)
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed
    )
    results = {}
    for method, cols in [("snapshot", len(VITALS)), ("temporal", x.shape[1])]:
        fit = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=12,
            min_samples_leaf=24,
            l2_regularization=3,
            random_state=seed,
        )
        fit.fit(x[train, :cols], y[train])
        p = fit.predict_proba(x[test, :cols])[:, 1]
        order = np.argsort(-p)[: len(test) // 10]
        results[method] = {
            **scores(y[test], p),
            "recall_top_10pct": float(y[test][order].sum() / y[test].sum()),
            "eligible_stays": len(y),
            "events": int(y.sum()),
        }
        save(fit, f"{method}-{seed}")
    return results


import csv
import io
import tarfile


def dataset(seed):
    x = []
    labels = []
    with tarfile.open(DATA / "icu2012/set-a.tar.gz") as z:
        for member in z:
            if not member.isfile() or not Path(member.name).stem.isdigit():
                continue
            early = {}
            future = []
            with io.TextIOWrapper(z.extractfile(member)) as stream:
                for row in csv.DictReader(stream):
                    name = row["Parameter"]
                    if name not in VITALS:
                        continue
                    h, m = map(int, row["Time"].split(":"))
                    stamp = h + m / 60
                    value = float(row["Value"])
                    if value <= 0:
                        continue
                    if name in VITALS[:2] and not 40 <= value <= 280:
                        continue
                    if stamp <= 6:
                        early.setdefault(name, []).append((stamp, value))
                    elif stamp <= 12 and name in VITALS[:2]:
                        future.append(value)
            previous = [v for k in VITALS[:2] for _, v in early.get(k, [])]
            if not previous or not future or min(previous) < 90:
                continue
            x.append(summarize(early))
            labels.append(int(min(future) < 90))
    return np.asarray(x), np.asarray(labels)


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
