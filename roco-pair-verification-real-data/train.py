"""Train an image-caption compatibility score on paired radiology embeddings."""

import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)


def metrics(scores):
    labels = np.eye(len(scores), dtype=bool)
    # All gallery negatives are included. Each genuine pair contributes one positive.
    return {
        "pair_auroc": roc_auc_score(labels.ravel(), scores.ravel()),
        "average_precision": average_precision_score(labels.ravel(), scores.ravel()),
        "recall_at_5": np.mean(
            [
                np.any(
                    np.argsort(-m, axis=1)[:, :5] == np.arange(len(m))[:, None], axis=1
                ).mean()
                for m in (scores, scores.T)
            ]
        ),
    }


class Compatibility:
    def __init__(self, images, captions, ridge):
        x, t = unit(images), unit(captions)
        self.mx, self.mt = x.mean(0), t.mean(0)
        xc, tc = x - self.mx, t - self.mt
        # Solve in the smaller sample space. Learn both directions from paired rows.
        self.xt = xc.T @ np.linalg.solve(xc @ xc.T + ridge * np.eye(len(x)), tc)
        self.tx = tc.T @ np.linalg.solve(tc @ tc.T + ridge * np.eye(len(t)), xc)

    def score(self, x, t):
        x, t = unit(x) - self.mx, unit(t) - self.mt
        return (unit(x @ self.xt) @ unit(t).T + unit(x) @ unit(t @ self.tx).T) / 2


def run(a, seed):
    x, t = a["train_image"], a["train_text"]
    fit, val = train_test_split(np.arange(len(x)), test_size=0.25, random_state=seed)
    cos = unit(x[val]) @ unit(t[val]).T
    candidates = []
    for ridge in (0.05, 0.5, 5.0):
        model = Compatibility(x[fit], t[fit], ridge)
        learned = model.score(x[val], t[val])
        for weight in (0.0, 0.1, 0.3, 0.6, 1.0):
            score = (1 - weight) * cos + weight * learned
            candidates.append((metrics(score)["average_precision"], weight, ridge))
    _, weight, ridge = max(candidates, key=lambda row: row[0])
    model = Compatibility(x, t, ridge)
    dest = ROOT / "checkpoints"
    dest.mkdir(exist_ok=True)
    np.savez_compressed(
        dest / f"blend-{seed}.npz",
        image_mean=model.mx,
        text_mean=model.mt,
        image_to_text=model.xt,
        text_to_image=model.tx,
        blend_weight=weight,
        ridge=ridge,
    )
    result = {}
    for tag, ik, tk in [
        ("clean", "test_image", "test_text"),
        ("blur", "test_blur_image", "test_text"),
        ("dropout", "test_image", "test_dropout_text"),
        ("both", "test_blur_image", "test_dropout_text"),
    ]:
        cos = unit(a[ik]) @ unit(a[tk]).T
        pred = model.score(a[ik], a[tk])
        for method, s in [
            ("cosine", cos),
            ("ridge", pred),
            ("blend", (1 - weight) * cos + weight * pred),
        ]:
            result[f"{method}_{tag}"] = metrics(s)
    return {"validation_weight": weight, "validation_ridge": ridge, "test": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data/roco_clip_embeddings.npz",
    )
    path = parser.parse_args().embeddings
    with np.load(path) as a:
        assert a["train_image"].shape == a["train_text"].shape == (200, 512)
        assert a["test_image"].shape == a["test_text"].shape == (80, 512)
        runs = {s: run(a, s) for s in [17, 29, 43, 59, 71]}
    means = {
        k: {m: np.mean([r["test"][k][m] for r in runs.values()]) for m in v}
        for k, v in runs[17]["test"].items()
    }
    (ROOT / "results.json").write_text(
        json.dumps(
            {
                "embedding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "training_pairs": 200,
                "test_pairs": 80,
                "runs": runs,
                "means": means,
            },
            indent=2,
        )
    )
