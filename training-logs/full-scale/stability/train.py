"""Compare internal and resampling-based selection of expression clusters."""

import argparse
import json
import tarfile
from itertools import islice
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]


def read_cellranger_counts(stream):
    banner = stream.readline()
    if banner.strip() != b"%%MatrixMarket matrix coordinate real general":
        raise ValueError("Unexpected Matrix Market representation")
    header = stream.readline()
    while header.startswith(b"%"):
        header = stream.readline()
    genes, cells, entries = map(int, header.split())
    counts = np.empty(entries, dtype=np.float32)
    indices = np.empty(entries, dtype=np.int32)
    per_cell = np.zeros(cells, dtype=np.int32)
    offset, previous_cell = 0, 0
    while block := b"".join(islice(stream, 100000)):
        fields = np.fromstring(
            block.decode("ascii"), dtype=np.float32, sep=" "
        ).reshape(-1, 3)
        rows = fields[:, 1].astype(np.int32) - 1
        columns = fields[:, 0].astype(np.int32) - 1
        if (
            rows[0] < previous_cell
            or (np.diff(rows) < 0).any()
            or rows[-1] >= cells
            or columns.min() < 0
            or columns.max() >= genes
            or not np.isfinite(fields[:, 2]).all()
            or (fields[:, 2] < 0).any()
        ):
            raise ValueError(
                "Cell Ranger coordinates or counts are out of order or invalid"
            )
        previous_cell = rows[-1]
        stop = offset + len(rows)
        indices[offset:stop] = columns
        counts[offset:stop] = fields[:, 2]
        per_cell += np.bincount(rows, minlength=cells).astype(np.int32)
        offset = stop
    if offset != entries:
        raise ValueError(f"Expected {entries} counts, read {offset}")
    pointer = np.r_[np.int32(0), np.cumsum(per_cell, dtype=np.int32)]
    matrix = sparse.csr_matrix(
        (counts, indices, pointer), shape=(cells, genes), copy=False
    )
    matrix.sum_duplicates()
    return matrix


def load_counts(seed, real):
    if real:
        with tarfile.open(ROOT / "datasets/pbmc68k-filtered.tar.gz", "r:gz") as archive:
            file = next(
                entry for entry in archive if entry.name.endswith("/matrix.mtx")
            )
            with archive.extractfile(file) as counts:
                matrix = read_cellranger_counts(counts)
        assert matrix.shape[0] >= 20000 and (matrix.data >= 0).all()
        return matrix, None
    rng = np.random.default_rng(seed)
    n, genes = 16000, 600
    labels = rng.choice(8, size=n, p=[0.2, 0.19, 0.16, 0.13, 0.11, 0.09, 0.07, 0.05])
    mean = np.full((8, genes), 0.07, dtype=np.float32)
    for label in range(8):
        mean[label, label * 26 : (label + 1) * 26] += rng.uniform(0.12, 0.8, size=26)
    batch = rng.integers(0, 4, n)
    batch_profile = rng.lognormal(0, 0.5, size=(4, genes))
    library = rng.lognormal(0, 0.5, size=(n, 1))
    counts = rng.poisson(mean[labels] * batch_profile[batch] * library).astype(
        np.float32
    )
    return sparse.csr_matrix(counts), labels


def normalize_counts(counts):
    library = np.asarray(counts.sum(axis=1)).ravel()
    factors = (10000 / np.maximum(1, library)).astype(np.float32)
    for row in range(0, len(library), 1024):
        end = min(row + 1024, len(library))
        part = slice(counts.indptr[row], counts.indptr[end])
        counts.data[part] *= np.repeat(
            factors[row:end], np.diff(counts.indptr[row : end + 1])
        )
        np.log1p(counts.data[part], out=counts.data[part])
    return counts


def cluster(kind, k, seed):
    if kind == "kmeans":
        return MiniBatchKMeans(
            n_clusters=k, max_iter=100, batch_size=1024, n_init=3, random_state=seed
        )
    return GaussianMixture(
        n_components=k,
        covariance_type="diag",
        max_iter=100,
        reg_covar=0.0005,
        random_state=seed,
    )


def features_and_models(matrix, train, valid, test, args):
    generator = np.random.default_rng(args.seed)
    train_counts = matrix[train]
    counts_mean = np.asarray(train_counts.mean(axis=0)).ravel()
    variety = np.asarray(train_counts.power(2).mean(axis=0)).ravel() - counts_mean**2
    eligible = np.flatnonzero(counts_mean > 0.01)
    genes = eligible[np.argsort(-variety[eligible], kind="stable")[:1800]]
    n = len(genes)
    dims = min(32, n - 1)
    full = TruncatedSVD(n_components=dims, n_iter=6, random_state=args.seed).fit(
        train_counts[:, genes]
    )
    tx, vx, hx = [full.transform(matrix[idx][:, genes]) for idx in (train, valid, test)]
    inspect = generator.choice(len(valid), min(1000, len(valid)), replace=False)
    candidates = [
        (kind, k) for kind in ("kmeans", "mixture") for k in (4, 6, 8, 10, 12)
    ]
    models, records, perturbations = {}, {}, {}
    for kind, k in candidates:
        key = f"{kind}-{k}"
        fitted = cluster(kind, k, args.seed).fit(tx)
        prediction = fitted.predict(vx)
        models[key] = fitted
        records[key] = {
            "algorithm": kind,
            "clusters": k,
            "validation_silhouette": float(
                silhouette_score(vx[inspect], prediction[inspect])
            ),
            "stability": [],
        }
        perturbations[key] = prediction
    for i in range(args.perturbations):
        sub_train = generator.choice(len(train), int(0.75 * len(train)), replace=False)
        keep = np.sort(generator.choice(n, int(n * 0.83), replace=False))
        projection = TruncatedSVD(
            n_components=dims, n_iter=5, random_state=args.seed + i + 1
        )
        train_subspace = projection.fit_transform(
            matrix[train[sub_train]][:, genes[keep]]
        )
        val_subspace = projection.transform(matrix[valid][:, genes[keep]])
        for kind, k in candidates:
            fitted = cluster(kind, k, args.seed + i + 1).fit(train_subspace)
            key = f"{kind}-{k}"
            observed = float(
                adjusted_rand_score(perturbations[key], fitted.predict(val_subspace))
            )
            records[key]["stability"].append(observed)
        print("Resampling replicate", i + 1, flush=True)
    return genes, full, models, records, hx, generator


def evaluate(
    output,
    names,
    models,
    records,
    transform,
    genes,
    matrix,
    test,
    reference,
    features,
    rng,
):
    inspect = rng.choice(len(test), min(1500, len(test)), replace=False)
    omitted = rng.choice(len(genes), int(0.22 * len(genes)), replace=False)
    keep = np.ones(len(genes), dtype=np.float32)
    keep[omitted] = 0
    dropout = matrix[test][:, genes].multiply(keep).tocsr()
    shifted = transform.transform(dropout)
    result = {}
    for method, key in names.items():
        model = models[key]
        predicted = model.predict(features)
        disturbed = model.predict(shifted)
        counts = np.bincount(predicted, minlength=records[key]["clusters"])
        result[method] = {
            "selected": key,
            "test_silhouette": float(
                silhouette_score(features[inspect], predicted[inspect])
            ),
            "test_dropout_agreement_ari": float(
                adjusted_rand_score(predicted, disturbed)
            ),
            "test_cluster_sizes": counts.tolist(),
        }
        if reference is not None:
            result[method]["test_truth_ari"] = float(
                adjusted_rand_score(reference[test], predicted)
            )
        joblib.dump(
            {"genes": genes, "svd": transform, "clusterer": model},
            output / f"{method}.joblib",
            compress=3,
        )
        np.savez_compressed(
            output / f"{method}-test.npz",
            row=test,
            clean=predicted,
            gene_dropout=disturbed,
        )
    return result


def main(args):
    counts, labels = load_counts(args.seed, args.real)
    matrix = normalize_counts(counts)
    del counts
    train_val, test = train_test_split(
        np.arange(matrix.shape[0]), test_size=0.2, random_state=args.seed
    )
    train, val = train_test_split(train_val, test_size=0.2, random_state=args.seed + 1)
    output = ROOT / (
        "real-data-projects/brock-pbmc-cluster-stability"
        if args.real
        else "professor-aligned/brock-expression-cluster-validation"
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "splits.npz", train=train, validation=val, test=test)
    genes, transform, fitted, records, features, rng = features_and_models(
        matrix, train, val, test, args
    )
    for record in records.values():
        record["stability_lower_quartile"] = float(
            np.quantile(record["stability"], 0.25)
        )
    names = {
        "silhouette": max(records, key=lambda k: records[k]["validation_silhouette"]),
        "stability": max(
            records,
            key=lambda k: (
                records[k]["stability_lower_quartile"],
                records[k]["validation_silhouette"],
            ),
        ),
    }
    result = {
        "configuration": vars(args),
        "cells": matrix.shape[0],
        "source_genes": matrix.shape[1],
        "selected_genes": len(genes),
        "splits": {"train": len(train), "validation": len(val), "test": len(test)},
        "candidates": records,
        "selected_test": evaluate(
            output,
            names,
            fitted,
            records,
            transform,
            genes,
            matrix,
            test,
            labels,
            features,
            rng,
        ),
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(result["selected_test"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--perturbations", type=int, default=5)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
