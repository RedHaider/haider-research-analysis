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


import csv
import io
import zipfile


def dataset(seed):
    x = []
    y = []
    groups = []
    keys = [
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
    ]
    with zipfile.ZipFile(DATA / "diabetes130.zip") as z:
        with z.open("diabetic_data.csv") as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw)):
                if row["discharge_disposition_id"] in {
                    "11",
                    "13",
                    "14",
                    "19",
                    "20",
                    "21",
                }:
                    continue
                x.append([float(row[k]) for k in keys])
                y.append(int(row["readmitted"] == "<30"))
                groups.append(int(row["patient_nbr"]))
    return np.array(x), np.array(y), np.array(groups)


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
