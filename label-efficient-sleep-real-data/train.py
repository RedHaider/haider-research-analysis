import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
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
    x = []
    y = []
    g = []
    for file in sorted((DATA / "sleep-accel/labels").glob("*sleep.txt")):
        pid = file.name.split("_")[0]
        hrfile = DATA / "sleep-accel/heart_rate" / (pid + "_heartrate.txt")
        if not hrfile.exists():
            continue
        hr = np.loadtxt(hrfile, delimiter=",")
        hr = hr[np.argsort(hr[:, 0])]
        for stamp, stage in np.loadtxt(file):
            if stage < 0:
                continue
            idx = np.searchsorted(hr[:, 0], [stamp - 300, stamp - 60, stamp + 30])
            long = hr[idx[0] : idx[2], 1]
            short = hr[idx[1] : idx[2], 1]
            if len(short) < 2 or len(long) < 4:
                continue
            x.append(
                [
                    short.mean(),
                    short.std(),
                    long.mean(),
                    long.std(),
                    short.mean() - long.mean(),
                    np.ptp(long),
                    np.median(long),
                    short[-1] - short[0],
                    len(short),
                    len(long),
                ]
            )
            y.append(int(stage != 0))
            g.append(pid)
    assert len(set(g)) >= 15
    return np.array(x), np.array(y), np.array(g)


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit


def run(seed):
    x, y, g = dataset(seed)
    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(x, y, g)
    )
    rng = np.random.default_rng(seed)
    labeled = rng.choice(tr, int(0.10 * len(tr)), replace=False)
    unlabeled = np.setdiff1d(tr, labeled)

    def fit(a, b):
        return ExtraTreesClassifier(
            n_estimators=96,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=2,
        ).fit(a, b)

    initial = fit(x[labeled], y[labeled])
    p = initial.predict_proba(x[unlabeled])[:, 1]
    accepted = np.abs(p - 0.5) > 0.42
    expanded = fit(
        np.concatenate([x[labeled], x[unlabeled][accepted]]),
        np.r_[y[labeled], p[accepted] >= 0.5],
    )
    output = {}
    for name, model in [("labeled_only", initial), ("pseudo_labels", expanded)]:
        prob = model.predict_proba(x[te])[:, 1]
        output[name] = {
            **scores(y[te], prob),
            "balanced_accuracy": float(balanced_accuracy_score(y[te], prob >= 0.5)),
            "test_subjects": len(np.unique(g[te])),
        }
    save((initial, expanded, np.unique(g[te]), int(accepted.sum())), f"models-{seed}")
    return output


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
