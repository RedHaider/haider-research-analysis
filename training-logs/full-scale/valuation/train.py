"""Value examples from bootstrap learners and measure the effect of pruning."""

import argparse
import gzip
import json
import time
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]


def load(real, seed):
    if real:
        with ZipFile(ROOT / "datasets/covertype.zip") as zip_file:
            name = next(
                name for name in zip_file.namelist() if name.endswith("covtype.data.gz")
            )
            with gzip.GzipFile(fileobj=zip_file.open(name)) as stream:
                matrix = pd.read_csv(stream, header=None, dtype=np.float32).to_numpy()
        x, y = matrix[:, :-1].copy(), matrix[:, -1].astype(int) - 1
        assert len(x) == 581012 and x.shape[1] == 54 and set(y) == set(range(7))
        groups = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
    else:
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(36000, 12))
        logits = np.column_stack(
            [
                latent[:, 0] * latent[:, 1] + 0.4 * latent[:, 2],
                latent[:, 3] ** 2 - 1.1,
                latent[:, 4] + 0.5 * latent[:, 5],
            ]
        ) + rng.normal(0, 0.25, (len(latent), 3))
        y = logits.argmax(1)
        x = np.column_stack(
            [
                latent,
                latent @ rng.normal(size=(12, 24)) + rng.normal(size=(len(latent), 24)),
                rng.normal(size=(len(latent), 12)),
            ]
        ).astype(np.float32)
        groups = np.arange(len(x))
    return x, y, groups


def splits(groups, seed):
    train_val, test = next(
        GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=seed).split(
            groups, groups=groups
        )
    )
    train, val = next(
        GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=seed + 1).split(
            train_val, groups=groups[train_val]
        )
    )
    result = train_val[train], train_val[val], test
    assert sum(len(np.unique(groups[index])) for index in result) == len(
        np.unique(groups)
    )
    return result


def corrupt_labels(x, labels, seed, rate):
    # Concentrate known flips along one measured-feature stratum.
    rng = np.random.default_rng(seed)
    priority = np.where(x[:, 0] > np.median(x[:, 0]), 3.0, 1.0)
    flips = rng.choice(
        len(labels), int(len(labels) * rate), replace=False, p=priority / priority.sum()
    )
    noisy = labels.copy()
    noisy[flips] = (
        noisy[flips] + rng.integers(1, len(np.unique(labels)), len(flips))
    ) % len(np.unique(labels))
    corrupted = noisy != labels
    return noisy, corrupted


def out_of_bag(forest, features, labels):
    n = len(labels)
    counts = np.zeros(n, dtype=np.int32)
    exact = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    for estimator, sampled in zip(forest.estimators_, forest.estimators_samples_):
        available = np.ones(n, dtype=bool)
        available[sampled] = False
        indexes = np.flatnonzero(available)
        probabilities = estimator.predict_proba(features[indexes])
        counts[indexes] += 1
        exact[indexes] += probabilities.argmax(1) == labels[indexes]
        confidence[indexes] += probabilities[np.arange(len(indexes)), labels[indexes]]
    assert (counts > 0).all()
    return exact / counts, confidence / counts, counts


def class_ranks(values, labels):
    result = np.empty(len(values))
    for label in np.unique(labels):
        mask = labels == label
        result[mask] = (
            pd.Series(values[mask]).rank(method="average", pct=True).to_numpy()
        )
    return result


def evaluate(model, x, y):
    prob = model.predict_proba(x)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prob.argmax(1))),
        "log_loss": float(log_loss(y, prob)),
        "macro_auroc": float(roc_auc_score(y, prob, multi_class="ovr")),
    }


def run(args):
    tag = (
        "real-data-projects/zou-covertype-data-valuation"
        if args.real
        else "professor-aligned/zou-out-of-bag-data-valuation"
    )
    out = ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    x, y, groups = load(args.real, args.seed)
    train, val, test = splits(groups, args.seed)
    np.savez_compressed(out / "splits.npz", train=train, validation=val, test=test)
    noisy, flips = corrupt_labels(x[train], y[train], args.seed + 2, 0.12)
    forest = ExtraTreesClassifier(
        n_estimators=args.trees,
        max_depth=20,
        min_samples_leaf=5,
        bootstrap=True,
        n_jobs=args.threads,
        random_state=args.seed + 3,
    )
    start = time.monotonic()
    forest.fit(x[train], noisy)
    exact, confidence, coverage = out_of_bag(forest, x[train], noisy)
    rankings = {
        "data_oob": exact,
        "class_normalized_oob_probability": class_ranks(confidence, noisy),
        "random": np.random.default_rng(args.seed + 4).random(len(train)),
    }
    np.savez_compressed(
        out / "valuations.npz",
        train=train,
        labels=noisy,
        flips=flips,
        counts=coverage,
        **rankings,
    )
    result = {
        "source_records": len(x),
        "features": x.shape[1],
        "configuration": vars(args),
        "training_flip_rate": float(flips.mean()),
        "split_counts": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "valuation": {},
        "retraining": {},
        "valuation_seconds": time.monotonic() - start,
    }
    cases = {"all_noisy": np.arange(len(train))}
    for name, values in rankings.items():
        order = np.lexsort((rankings["random"], values))
        remove = int(0.12 * len(train))
        cases[name] = order[remove:]
        result["valuation"][name] = {
            "flip_average_precision": float(average_precision_score(flips, -values)),
            "flip_precision_at_pruning_budget": float(flips[order[:remove]].mean()),
            "pruned_per_observed_class": np.bincount(
                noisy[order[:remove]], minlength=len(np.unique(y))
            ).tolist(),
        }
    print("Valuation results", result["valuation"], flush=True)
    for name, keep in cases.items():
        model = HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=48,
            learning_rate=0.09,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.seed + 5,
        )
        model.fit(x[train[keep]], noisy[keep])
        joblib.dump(model, out / f"{name}.joblib", compress=3)
        result["retraining"][name] = {
            "size": len(keep),
            "validation": evaluate(model, x[val], y[val]),
            "test": evaluate(model, x[test], y[test]),
        }
        (out / "progress.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(name, result["retraining"][name], flush=True)
    (out / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--trees", type=int, default=96)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=27)
    arguments = parser.parse_args()
    with threadpool_limits(limits=arguments.threads):
        run(arguments)
