import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
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


import cv2
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

cv2.setNumThreads(1)


def center(mask):
    y, x = np.where(mask)
    return np.array([x.mean(), y.mean()])


def features(a, b, mask):
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 19, 3, 5, 1.2, 0)
    inside = flow[mask]
    outside = flow[~mask]
    local = np.median(inside, axis=0)
    shift, _ = cv2.phaseCorrelate(a.astype("float32"), b.astype("float32"))
    return np.r_[
        local,
        inside.mean(0),
        np.quantile(inside, 0.25, axis=0),
        np.quantile(inside, 0.75, axis=0),
        np.median(outside, axis=0),
        shift,
    ]


def example(rows, a, b, m, n, group):
    if 30 <= m.sum() < m.size * 0.9 and n.sum() >= 30:
        rows.append((features(a, b, m), center(n) - center(m), group))


def public():
    root = DATA / "kvasir-seg" / "extracted" / "Kvasir-SEG"
    paths = sorted((root / "images").glob("*.jpg"))
    assert len(paths) == 1000
    rng = np.random.default_rng(117)
    selected = np.sort(rng.choice(len(paths), 420, replace=False))
    rows = []
    for index in selected:
        path = paths[index]
        a = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (160, 96))
        m = (
            cv2.resize(
                cv2.imread(str(root / "masks" / path.name), cv2.IMREAD_GRAYSCALE),
                (160, 96),
                interpolation=cv2.INTER_NEAREST,
            )
            > 127
        )
        for _ in range(3):
            x, y = (float(z) for z in center(m))
            mat = cv2.getRotationMatrix2D((x, y), rng.normal(0, 8), 1.0)
            mat[:, 0] *= rng.uniform(0.87, 1.13)
            mat[:, 2] += rng.normal(0, 5, 2)
            b = cv2.warpAffine(a, mat, (160, 96), borderMode=cv2.BORDER_REFLECT)
            b = np.clip(
                b * rng.uniform(0.68, 1.25) + rng.normal(0, 6, b.shape), 0, 255
            ).astype("uint8")
            n = (
                cv2.warpAffine(
                    m.astype("uint8"), mat, (160, 96), flags=cv2.INTER_NEAREST
                )
                > 0
            )
            example(rows, a, b, m, n, path.stem)
    assert len(rows) >= 800
    return rows


def run(seed):
    x, y, g = observations
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.28, random_state=seed).split(x, y, g)
    )
    assert not set(g[train]) & set(g[test])
    model = make_pipeline(StandardScaler(), Ridge(alpha=18)).fit(x[train], y[train])
    save((model, np.unique(g[test])), f"ridge-{seed}")
    return {
        method: {
            "mean_error_px": float(np.linalg.norm(pred - y[test], axis=1).mean()),
            "median_error_px": float(np.median(np.linalg.norm(pred - y[test], axis=1))),
            "test_sequences": len(np.unique(g[test])),
            "test_pairs": len(test),
        }
        for method, pred in [
            ("stationary", np.zeros_like(y[test])),
            ("median_flow", x[test, :2]),
            ("learned_fusion", model.predict(x[test])),
        ]
    }


if __name__ == "__main__":
    rows = public()
    x, y, g = zip(*rows)
    observations = (np.stack(x), np.stack(y), np.array(g))
    runs = {seed: run(seed) for seed in SEEDS}
    means = {
        method: {
            key: float(np.mean([run[method][key] for run in runs.values()]))
            for key in metrics
        }
        for method, metrics in runs[SEEDS[0]].items()
    }
    (ROOT / "results.json").write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "runs": runs,
                "means": means,
                "samples": len(g),
                "sequences": len(set(g)),
            },
            indent=2,
        )
    )
    print(means)
