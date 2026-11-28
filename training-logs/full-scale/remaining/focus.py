import numpy as np
import pandas as pd
from common import DATA, save, split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


def prepare(real):
    if real:
        table = pd.read_csv(DATA / "bbbc006-focus.csv")
        step = table["Image_Metadata_ZStep"].to_numpy()
        group = table["Image_Metadata_Well"].astype(str).to_numpy()
        keys = [name for name in table if name.startswith("Image_ImageQuality_")]
        x = table[keys].to_numpy()
        index = next(i for i, k in enumerate(keys) if "FocusScore_OrigDAPI" in k)
        return x, (step >= 11) & (step <= 23), group, step, index
    rng = np.random.default_rng(91)
    groups = np.repeat(np.arange(650), 34)
    step = np.tile(np.arange(34), 650)
    depth = abs(step - 17)
    brightness = rng.lognormal(0, 0.6, 650)[groups]
    factor = np.exp(-depth / 11)
    x = np.column_stack(
        [
            brightness * factor + rng.normal(0, 0.2, len(depth)),
            factor + rng.normal(0, 0.11, len(depth)),
            np.exp(-depth / 5) + rng.normal(0, 0.2, len(depth)),
            1 / (1 + depth) + rng.normal(0, 0.05, len(depth)),
            brightness,
        ]
    )
    return x, (depth <= 6), groups, step, 0


def run(real):
    x, y, group, depth, focus = prepare(real)
    assert len(np.unique(y)) == 2
    train, test = split(y, group)
    models, score, result = {}, {}, {}
    for name, idx in (
        ("single_channel_focus", slice(focus, focus + 1)),
        ("widefield_focus_fusion", slice(None)),
    ):
        clf = HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=27, min_samples_leaf=25, random_state=91
        )
        clf.fit(x[train, idx], y[train])
        p = clf.predict_proba(x[test, idx])[:, 1]
        models[name], score[name] = clf, p
        result[name] = {
            "auroc": float(roc_auc_score(y[test], p)),
            "ap": float(average_precision_score(y[test], p)),
        }
    save(
        "dickensheets-bbbc-microscope-focus"
        if real
        else "dickensheets-optical-section-confidence",
        real,
        {
            "rows": len(y),
            "held_out_groups": len(np.unique(group[test])),
            "source": "BBBC006 measured image quality features"
            if real
            else "simulated defocus profiles",
            "metrics": result,
        },
        models,
        {
            "focus_label": y[test],
            "z_step": depth[test],
            "train_groups": group[train],
            "test_groups": group[test],
            **score,
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            run(real)
