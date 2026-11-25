import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", str(ROOT / "data")))
SEEDS = [13, 37, 61]


def save(model, name):
    path = ROOT / "checkpoints"
    path.mkdir(exist_ok=True)
    joblib.dump(model, path / (name + ".joblib"), compress=3)


from sklearn.impute import SimpleImputer


def design(x, stop, event):
    cuts = np.array([90, 180, 270, 365, 550, 730])
    rows = []
    labels = []
    for start, end in zip([0, *cuts[:-1]], cuts):
        use = (stop > start) & ((stop >= end) | event)
        rows.extend(np.column_stack([x[use], np.full(use.sum(), np.log1p(start))]))
        labels.extend(event[use] & (stop[use] <= end))
    return np.array(rows), np.array(labels)


def run(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2600, 7))
    event = rng.exponential(450 * np.exp(-0.55 * x[:, 0] - 0.8 * x[:, 4]))
    cens = rng.exponential(750, size=len(x))
    t = np.minimum(event, cens)
    e = event <= cens
    tr, te = train_test_split(
        np.arange(len(x)), test_size=0.30, stratify=e, random_state=seed
    )

    def masked(a):
        a = a.copy()
        a[:, 4:] = np.nan
        return a

    out = {}
    for name in ["complete_training", "view_dropout"]:
        a = x[tr]
        times = t[tr]
        events = e[tr]
        if name == "view_dropout":
            a = np.vstack([a, masked(a)])
            times = np.tile(times, 2)
            events = np.tile(events, 2)
        features, target = design(a, times, events)
        model = make_pipeline(
            SimpleImputer(add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.15, max_iter=500),
        ).fit(features, target)
        save(model, f"{name}-{seed}")
        for condition, z in [("available", x[te]), ("missing", masked(x[te]))]:
            hazard = np.array(
                [
                    model.predict_proba(np.c_[z, np.full(len(z), np.log1p(b))])[:, 1]
                    for b in [0, 90, 180, 270]
                ]
            )
            risk = 1 - (1 - hazard).prod(0)
            comparable = (t[te, None] < t[te][None, :]) & e[te, None]
            diff = risk[:, None] - risk
            out[name + "_" + condition] = {
                "concordance": float(
                    ((diff > 0) + 0.5 * (diff == 0))[comparable].mean()
                ),
                "comparable_pairs": int(comparable.sum()),
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
