import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
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


def dataset(seed):
    with np.load(DATA / "medmnist/pneumoniamnist.npz") as f:
        return (
            f["train_images"].astype(np.float32) / 255,
            f["train_labels"].ravel(),
            f["test_images"].astype(np.float32) / 255,
            f["test_labels"].ravel(),
        )


import scipy.ndimage as ndi
from scipy.special import expit
from sklearn.neural_network import MLPClassifier


def run(seed):
    x, y, z, t = dataset(seed)
    a, b = [
        ndi.zoom(v, (1, 12 / v.shape[1], 12 / v.shape[2]), order=1).reshape(len(v), -1)
        for v in (x, z)
    ]
    scaler = StandardScaler().fit(a)
    model = MLPClassifier(
        hidden_layer_sizes=(48, 16), max_iter=75, early_stopping=True, random_state=seed
    ).fit(scaler.transform(a), y)
    save((scaler, model), f"fp-model-{seed}")
    u = scaler.transform(b)
    base = model.predict_proba(u)[:, 1]
    arrays = list(zip(model.coefs_, model.intercepts_))
    out = {
        "fp32": {
            **scores(t, base),
            "parameter_payload_bytes": sum(4 * (w.size + c.size) for w, c in arrays),
            "disagreement": 0.0,
        }
    }
    for bits in (8, 4):
        qmax = 2 ** (bits - 1) - 1
        quant = []
        v = u
        for i, (w, c) in enumerate(arrays):
            factor = np.maximum(np.abs(w).max(0), 1e-12) / qmax
            code = np.rint(w / factor).clip(-qmax, qmax).astype(np.int8)
            quant.append((code, factor, c))
            v = v @ (code * factor) + c
            if i < len(arrays) - 1:
                v = np.maximum(v, 0)
        p = expit(v.ravel())
        save((scaler, quant, bits), f"quant-{bits}-{seed}")
        out[f"int{bits}_weights"] = {
            **scores(t, p),
            "parameter_payload_bytes": sum(
                (w.size * bits + 7) // 8 + 4 * (f.size + c.size) for w, f, c in quant
            ),
            "disagreement": float(np.mean((p >= 0.5) != (base >= 0.5))),
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
