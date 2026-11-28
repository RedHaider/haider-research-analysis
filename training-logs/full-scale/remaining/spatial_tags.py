import io

import numpy as np
from common import DATA, classification, save, split
from PIL import Image
from pyarrow.parquet import ParquetFile
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def features(images, grid):
    _, h, w, _ = images.shape
    a = images.astype(np.float32) / 255
    parts = []
    for i in range(grid):
        for j in range(grid):
            p = a[
                :,
                i * h // grid : (i + 1) * h // grid,
                j * w // grid : (j + 1) * w // grid,
            ]
            gx = np.abs(np.diff(p, axis=2))
            gy = np.abs(np.diff(p, axis=1))
            parts.extend(
                [p.mean((1, 2)), p.std((1, 2)), gx.mean((1, 2)), gy.mean((1, 2))]
            )
            for cut in (0.2, 0.4, 0.6, 0.8):
                parts.append((p < cut).mean((1, 2), dtype=np.float32))
    return np.column_stack(parts)


def read_cifar(part):
    whole, regions, labels = [], [], []
    source = ParquetFile(DATA / f"cifar100-{part}.parquet")
    for batch in source.iter_batches(batch_size=256, columns=["img", "coarse_label"]):
        images = []
        for row in batch.to_pylist():
            with Image.open(io.BytesIO(row["img"]["bytes"])) as img:
                images.append(np.asarray(img.convert("RGB")))
            labels.append(row["coarse_label"])
        images = np.stack(images)
        whole.append(features(images, 1))
        regions.append(features(images, 3))
    return np.concatenate(whole), np.concatenate(regions), np.asarray(labels)


def simulation():
    rng = np.random.default_rng(325)
    labels = np.arange(16000) % 8
    xy = np.mgrid[:30, :30]
    palette = np.array([[160, 75, 55], [50, 130, 165], [125, 162, 75], [110, 88, 150]])
    images = np.empty((len(labels), 30, 30, 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        quadrant = (xy[0] >= 15 + rng.integers(-3, 4)) * 2 + (
            xy[1] >= 15 + rng.integers(-3, 4)
        )
        order = (np.arange(4) * (1 if label < 4 else -1) + label % 4) % 4
        image = (
            palette[order[quadrant]]
            + rng.normal(0, 27, (30, 30, 3))
            + rng.normal(0, 25, (1, 1, 3))
        )
        images[i] = np.clip(image, 0, 255)
    return features(images, 1), features(images, 3), labels


def main(real):
    if real:
        a, b, y = read_cifar("train")
        c, d, target = read_cifar("test")
        train, test = np.arange(len(y)), np.arange(len(target))
    else:
        whole, spatial, labels = simulation()
        train, test = split(labels)
        a, b, y = whole[train], spatial[train], labels[train]
        c, d, target = whole[test], spatial[test], labels[test]
    report = {
        "source": "CIFAR-100 coarse labels" if real else "synthetic spatial palettes",
        "training_images": len(y),
        "test_images": len(target),
        "arms": {},
    }
    models, predictions = {}, {}
    for name, x, x_test in (("global", a, c), ("spatial_region", b, d)):
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(192, 128),
                max_iter=35,
                early_stopping=True,
                batch_size=256,
                random_state=23,
            ),
        )
        model.fit(x, y)
        score = model.predict_proba(x_test)
        report["arms"][name] = {
            **classification(target, score, model.classes_),
            "top_5": float(
                (model.classes_[np.argsort(score, axis=1)[:, -5:]] == target[:, None])
                .any(1)
                .mean()
            ),
            "parameters": int(
                sum(w.size for w in model[-1].coefs_)
                + sum(w.size for w in model[-1].intercepts_)
            ),
        }
        models[name], predictions[name] = model, score
    save(
        "li-cifar-semantic-regions" if real else "li-spatial-annotation",
        real,
        report,
        models,
        {"targets": target, "train_index": train, "test_index": test, **predictions},
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
