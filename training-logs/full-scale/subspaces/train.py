"""Compare anchor-based sparse self-expression with cosine graphs and k-means."""

import argparse
import gzip
import json
import struct
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]


def read_images(path):
    with gzip.open(path, "rb") as stream:
        magic, length, height, width = struct.unpack(">IIII", stream.read(16))
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    assert magic == 2051 and len(pixels) == length * height * width
    return pixels.reshape(length, height * width).astype(np.float32) / 255


def read_labels(path):
    with gzip.open(path, "rb") as stream:
        magic, length = struct.unpack(">II", stream.read(8))
        labels = np.frombuffer(stream.read(), dtype=np.uint8).astype(int)
    assert magic == 2049 and len(labels) == length
    return labels


def data(real, seed):
    if real:
        directory = ROOT / "datasets/fashion-mnist"
        parts = [
            (
                read_images(directory / f"{split}-images-idx3-ubyte.gz"),
                read_labels(directory / f"{split}-labels-idx1-ubyte.gz"),
            )
            for split in ("train", "t10k")
        ]
        assert [len(x) for x, _ in parts] == [60000, 10000]
        return (*parts[0], *parts[1])
    rng = np.random.default_rng(seed)
    shared, _ = np.linalg.qr(rng.normal(size=(96, 3)))
    basis = [np.linalg.qr(rng.normal(size=(96, 7)))[0] for _ in range(8)]
    labels = np.repeat(np.arange(len(basis)), 2000)
    values = np.empty((len(labels), 96), dtype=np.float32)
    for index, directions in enumerate(basis):
        use = labels == index
        signal = rng.normal(size=(use.sum(), 7)) @ directions.T
        common = rng.normal(size=(use.sum(), 3)) @ shared.T
        values[use] = signal + common + rng.normal(0, 0.08, signal.shape)
    train, test = train_test_split(
        np.arange(len(labels)), test_size=0.25, random_state=seed, stratify=labels
    )
    return values[train], labels[train], values[test], labels[test]


def affinity(queries, dictionary, mode, width=32, sparsity=6):
    rows, columns, weights = [], [], []
    for start in range(0, len(queries), 256):
        query = queries[start : start + 256]
        dots = query @ dictionary.T
        choice = np.argpartition(-np.abs(dots), width - 1, axis=1)[:, :width]
        atoms = dictionary[choice]
        if mode == "cosine_knn":
            relevance = np.take_along_axis(dots, choice, axis=1)
            support = np.argpartition(-np.abs(relevance), sparsity - 1, axis=1)[
                :, :sparsity
            ]
            coeff = np.abs(np.take_along_axis(relevance, support, axis=1))
        else:
            sample = np.arange(len(query))
            support = np.empty((len(query), sparsity), dtype=int)
            residual = query.copy()
            for step in range(sparsity):
                relevance = np.abs(np.einsum("bad,bd->ba", atoms, residual))
                relevance[sample[:, None], support[:, :step]] = -1
                support[:, step] = relevance.argmax(axis=1)
                active = atoms[sample[:, None], support[:, : step + 1]]
                gram = active @ active.transpose(0, 2, 1)
                gram += np.eye(step + 1, dtype=gram.dtype) * 1e-4
                target = active @ query[:, :, None]
                coeff = np.linalg.solve(gram, target).squeeze(-1)
                residual = query - np.einsum("ba,bad->bd", coeff, active)
            coeff = np.abs(coeff)
        selected = np.take_along_axis(choice, support, axis=1)
        rows.append(np.repeat(np.arange(start, start + len(query)), sparsity))
        columns.append(selected.ravel())
        weights.append((coeff / np.maximum(coeff.sum(1, keepdims=True), 1e-8)).ravel())
    return csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(queries), len(dictionary)),
    )


def metrics(y, predicted, mapping):
    return {
        "ari": float(adjusted_rand_score(y, predicted)),
        "nmi": float(normalized_mutual_info_score(y, predicted)),
        "training_mapped_accuracy": float((y == mapping[predicted]).mean()),
    }


def record(model, train, y_train, test, y_test, shifted, clusters):
    train_pred = model.predict(train)
    idx_class, idx_cluster = linear_sum_assignment(
        -confusion_matrix(y_train, train_pred, labels=np.arange(clusters))
    )
    mapping = np.empty(clusters, dtype=int)
    mapping[idx_cluster] = idx_class
    return {
        "training": metrics(y_train, train_pred, mapping),
        "clean_test": metrics(y_test, model.predict(test), mapping),
        "perturbed_test": metrics(y_test, model.predict(shifted), mapping),
    }, mapping


def main(args):
    rng = np.random.default_rng(args.seed)
    x, y, future, targets = data(args.real, args.seed)
    output = ROOT / (
        "real-data-projects/vidal-fashion-anchor-subspaces"
        if args.real
        else "professor-aligned/vidal-sparse-subspace-recovery"
    )
    output.mkdir(parents=True, exist_ok=True)
    disturbed = future.copy()
    if args.real:
        shaped = disturbed.reshape(-1, 28, 28)
        for image, row in zip(shaped, rng.integers(0, 22, size=len(shaped))):
            image[row : row + 7] = 0
        disturbed = np.clip(
            disturbed + rng.normal(0, 0.08, disturbed.shape), 0, 1
        ).astype(np.float32)
    else:
        disturbed += rng.normal(0, 0.20, disturbed.shape).astype(np.float32)
    reduce = PCA(n_components=64, random_state=args.seed, svd_solver="randomized").fit(
        x
    )
    train, test, shifted = [
        normalize(reduce.transform(values).astype(np.float32))
        for values in (x, future, disturbed)
    ]
    k = len(np.unique(y))
    estimator = MiniBatchKMeans(
        n_clusters=k, n_init=10, batch_size=1024, random_state=args.seed
    )
    results = {
        "train_records": len(x),
        "test_records": len(future),
        "configuration": vars(args),
        "arms": {},
    }
    raw = estimator.fit(train)
    results["arms"]["pca_kmeans"], mapping = record(
        raw, train, y, test, targets, shifted, k
    )
    joblib.dump(
        {"pca": reduce, "kmeans": raw, "mapping": mapping},
        output / "pca_kmeans.joblib",
        compress=3,
    )
    landmarks = train[rng.choice(len(train), args.anchors, replace=False)]
    for name in ("cosine_knn", "sparse_omp"):
        start = time.monotonic()
        graph = affinity(train, landmarks, name)
        degree = 1 / np.sqrt(np.maximum(np.asarray(graph.sum(axis=0)).ravel(), 1e-8))
        _, singular, vectors = randomized_svd(
            graph.multiply(degree).tocsr(),
            n_components=k,
            n_iter=8,
            random_state=args.seed,
        )
        transform = (degree[:, None] * vectors.T) / np.maximum(singular, 1e-8)
        embedding = normalize(graph @ transform)
        model = MiniBatchKMeans(
            n_clusters=k, n_init=10, batch_size=1024, random_state=args.seed
        ).fit(embedding)
        test_embedding = normalize(affinity(test, landmarks, name) @ transform)
        shifted_embedding = normalize(affinity(shifted, landmarks, name) @ transform)
        result, mapping = record(
            model, embedding, y, test_embedding, targets, shifted_embedding, k
        )
        result["fit_and_evaluation_seconds"] = time.monotonic() - start
        results["arms"][name] = result
        joblib.dump(
            {
                "pca": reduce,
                "landmarks": landmarks,
                "transform": transform,
                "kmeans": model,
                "mapping": mapping,
            },
            output / f"{name}.joblib",
            compress=3,
        )
        (output / "progress.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(name, result, flush=True)
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--anchors", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        main(args)
