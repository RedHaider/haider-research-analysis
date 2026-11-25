import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
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


import edfio
from scipy.signal import butter, sosfiltfilt


def dataset(seed):
    x = []
    y = []
    g = []
    labels = "fc5 fc3 fc1 c5 c3 c1 cp5 cp3 cp1 fc6 fc4 fc2 c6 c4 c2 cp6 cp4 cp2".split()
    for pid in range(1, 17):
        for run_id in [4, 8]:
            record = edfio.read_edf(DATA / "eegmmidb" / f"S{pid:03}R{run_id:02}.edf")
            signals = {s.label.strip(".").lower(): s.data for s in record.signals}
            a = np.stack([signals[k] for k in labels])
            # Reference each hemisphere independently to avoid contamination across masked views.
            for side in [a[:9], a[9:]]:
                side -= side.mean(0)
            bands = [
                sosfiltfilt(butter(4, b, btype="bandpass", fs=160, output="sos"), a)
                for b in [(7, 12), (12, 20), (20, 30)]
            ]
            for event in record.annotations:
                if event.text not in ("T1", "T2"):
                    continue
                j = round((event.onset + 0.5) * 160)
                x.append(
                    np.stack(
                        [np.log(b[:, j : j + 480].var(1) + 1e-12) for b in bands], 1
                    ).ravel()
                )
                y.append(int(event.text == "T2"))
                g.append(pid)
    return np.stack(x), np.array(y), np.array(g)


from sklearn.model_selection import GroupShuffleSplit


def run(seed):
    x, y, g = dataset(seed)
    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(x, y, g)
    )
    mean = x[tr].mean(0)
    std = x[tr].std(0) + 1e-6
    a = (x[tr] - mean) / std
    b = (x[te] - mean) / std

    def missing(z):
        z = z.copy()
        z[:, z.shape[1] // 2 :] = 0
        return z

    out = {}
    for name in ["standard", "dropout"]:
        f = a
        target = y[tr]
        if name == "dropout":
            f = np.concatenate([a, missing(a)])
            target = np.tile(target, 2)
        model = LogisticRegression(C=0.08, max_iter=500, class_weight="balanced").fit(
            f, target
        )
        for cond, z in [("intact", b), ("missing", missing(b))]:
            out[name + "_" + cond] = scores(y[te], model.predict_proba(z)[:, 1])
        save((mean, std, model, np.unique(g[te])), f"{name}-{seed}")
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
