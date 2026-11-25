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
    rng = np.random.default_rng(seed)
    x = []
    y = []
    g = []
    for person in range(40):
        wake = True
        offset = rng.normal(0, 5)
        history = []
        for epoch in range(400):
            if rng.random() < (0.07 if wake else 0.025):
                wake = not wake
            rate = 65 + offset + 7 * wake + rng.normal(0, 4)
            history.append(rate)
            window = np.array(history[-10:])
            x.append(
                [
                    rate,
                    window.mean(),
                    window.std(),
                    np.ptp(window),
                    rate - window.mean(),
                ]
            )
            y.append(int(not wake))
            g.append(person)
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
