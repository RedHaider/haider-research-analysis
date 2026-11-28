import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets"


def save(slug, real, report, models, arrays):
    folder = ROOT / ("real-data-projects" if real else "professor-aligned") / slug
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, folder / "models.joblib", compress=3)
    safe_arrays = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        safe_arrays[name] = array.astype(str) if array.dtype.kind == "O" else array
    np.savez_compressed(folder / "held-out.npz", **safe_arrays)
    staged = folder / "results.tmp"
    staged.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    staged.replace(folder / "results.json")
    print(slug, json.dumps(report, default=str)[:1200], flush=True)


def split(y, groups=None, seed=2027):
    ids = np.arange(len(y))
    if groups is None:
        return train_test_split(ids, test_size=0.22, stratify=y, random_state=seed)
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=seed).split(
            ids, groups=groups
        )
    )
    assert not set(groups[train]) & set(groups[test])
    return train, test


def classification(y, predicted, classes=None):
    y = np.asarray(y)
    result = {
        "accuracy": float(
            accuracy_score(
                y, predicted if classes is None else classes[predicted.argmax(axis=1)]
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                predicted if classes is None else classes[predicted.argmax(axis=1)],
                average="macro",
                zero_division=0,
            )
        ),
    }
    if classes is not None and len(classes) == len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y == classes[1], predicted[:, 1]))
    return result
