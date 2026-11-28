"""Benchmark budget-matched search with temporal validation."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Trial:
    learner: str
    iterations: int
    capacity: int
    tuning: float

    def build(self, seed):
        if self.learner == "histogram":
            return HistGradientBoostingClassifier(
                max_iter=self.iterations,
                max_leaf_nodes=self.capacity,
                learning_rate=self.tuning,
                l2_regularization=1,
                early_stopping=False,
                random_state=seed,
            )
        return ExtraTreesClassifier(
            n_estimators=self.iterations,
            max_depth=self.capacity,
            max_features=self.tuning,
            min_samples_leaf=4,
            n_jobs=1,
            random_state=seed,
        )

    def cost(self, n):
        capacity_scale = self.capacity / (16 if self.learner == "histogram" else 8)
        return int(n * self.iterations * capacity_scale)


def real_data():
    rows, _ = arff.loadarff(str(ROOT / "datasets/electricity-151.arff"))
    raw = pd.DataFrame(rows)
    assert len(raw) == 45312
    timestamp = ["date", "period"]
    raw = (
        raw.drop_duplicates(subset=timestamp, keep=False)
        .sort_values(timestamp, kind="stable")
        .reset_index(drop=True)
    )
    assert all(raw["class"].isin([b"UP", b"DOWN"]))
    # Classify the next available record using calendar data and previous observations.
    frame = pd.DataFrame(index=raw.index)
    period = raw["period"].to_numpy() * 47 / 48
    for x, cycle in ((period, "period"), ((raw.day.astype(int) - 1) / 7, "weekday")):
        frame[f"{cycle}_sin"] = np.sin(2 * np.pi * x)
        frame[f"{cycle}_cos"] = np.cos(2 * np.pi * x)
    for name in ("nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"):
        past = raw[name].shift(1)
        for window in (1, 3, 12, 48, 192):
            frame[f"{name}_lag_{window}"] = raw[name].shift(window)
        for window in (3, 12, 48, 192):
            frame[f"{name}_mean_{window}"] = past.rolling(window).mean()
            frame[f"{name}_std_{window}"] = past.rolling(window).std(ddof=0)
    return frame.iloc[192:].to_numpy(dtype=np.float32), (
        raw["class"].iloc[192:] == b"UP"
    ).to_numpy(dtype=int)


def synthetic_data(seed):
    rng = np.random.default_rng(seed)
    n = 48000
    drift = np.repeat(np.array([0, 0.5, -0.5, 0.3, 0.2, -0.2, 0.6, 0.1]), n // 8)
    features = rng.normal(size=(n, 28)).astype(np.float32)
    logits = (
        1.1 * features[:, 0] * features[:, 1]
        + np.sin(1.7 * features[:, 3])
        - 0.9 * features[:, 4] ** 2
        + drift
    )
    logits += (0.6 + drift) * features[:, 5] + rng.normal(0, 0.35, n)
    labels = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    return features, labels


def candidates(seed):
    rng = np.random.default_rng(seed)
    trials = []
    for learner, capacities, tunings in (
        ("histogram", (15, 25, 39, 63), (0.04, 0.08, 0.13, 0.20)),
        ("extra_trees", (7, 10, 14, 19), (0.4, 0.6, 0.8, 1.0)),
    ):
        for iterations in (35, 60, 100):
            for capacity in capacities:
                trials.append(
                    Trial(learner, iterations, capacity, float(rng.choice(tunings)))
                )
    return [trials[i] for i in rng.permutation(len(trials))]


def evaluate(y, prob):
    return {
        "average_precision": float(average_precision_score(y, prob)),
        "auroc": float(roc_auc_score(y, prob)),
        "log_loss": float(log_loss(y, prob)),
    }


def search(x, y, train, validation, options, seed, budget, mode):
    records, incumbent = [], None
    consumed = 0
    fractions = (1.0,) if mode == "random_full" else (0.25, 0.5, 1.0)
    active = options
    for fraction in fractions:
        n = int(fraction * len(train))
        n_train = train[-n:]
        remaining = budget - consumed
        # Preserve part of the resource budget for later promotions.
        limit = remaining if fraction == 1 else int(remaining * 0.42)
        used = 0
        scored = []
        for trial in active:
            cost = trial.cost(n)
            if cost + used > limit:
                continue
            model = trial.build(seed)
            start = time.process_time()
            model.fit(x[n_train], y[n_train])
            probabilities = model.predict_proba(x[validation])[:, 1]
            seconds = time.process_time() - start
            result = evaluate(y[validation], probabilities)
            scored.append((result["average_precision"], trial))
            used += cost
            records.append(
                {
                    "trial": asdict(trial),
                    "fraction": fraction,
                    "cpu_seconds": seconds,
                    "units": cost,
                    "cumulative_units": consumed + used,
                    "validation": result,
                }
            )
            if fraction == 1 and (
                incumbent is None or result["average_precision"] > incumbent[0]
            ):
                incumbent = (result["average_precision"], trial)
        consumed += used
        active = [
            trial
            for _, trial in sorted(scored, key=lambda item: item[0], reverse=True)[
                : max(3, len(scored) // 2)
            ]
        ]
    if incumbent is None:
        raise RuntimeError(f"{mode} never reached a full-data trial")
    return incumbent[1], records, consumed


def write(path, value):
    part = path.with_suffix(".tmp")
    part.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


def main(args):
    x, y = real_data() if args.real else synthetic_data(args.seed)
    t, v = int(0.60 * len(y)), int(0.80 * len(y))
    gap = 192 if args.real else 0
    train, validation, test = (
        np.arange(t - gap),
        np.arange(t, v - gap),
        np.arange(v, len(y)),
    )
    budget = args.budget_multiple * len(train) * 100
    prefix = (
        "real-data-projects/wu-electricity-frugal-search"
        if args.real
        else "professor-aligned/wu-fidelity-automl"
    )
    output = ROOT / prefix
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "splits.npz", train=train, validation=validation, test=test
    )
    options = candidates(args.seed)
    results = {
        "configuration": vars(args),
        "source_rows_used": len(y),
        "features": x.shape[1],
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "search_budget_per_arm": budget,
        "methods": {},
    }
    for mode in ("random_full", "successive_fidelity"):
        winner, log, consumed = search(
            x, y, train, validation, options, args.seed, budget, mode
        )
        start = time.process_time()
        final = winner.build(args.seed)
        full_training = np.r_[train, validation]
        final.fit(x[full_training], y[full_training])
        elapsed = time.process_time() - start
        prob = final.predict_proba(x[test])[:, 1]
        result = {
            "winner": asdict(winner),
            "units_spent": consumed,
            "search_cpu_seconds": sum(r["cpu_seconds"] for r in log),
            "refit_cpu_seconds": elapsed,
            "test": evaluate(y[test], prob),
        }
        results["methods"][mode] = result
        write(output / f"trials-{mode}.json", log)
        joblib.dump(final, output / f"{mode}.joblib", compress=3)
        np.savez_compressed(
            output / f"test-{mode}.npz", predictions=prob, label=y[test], index=test
        )
        write(output / "progress.json", results)
        print(mode, result, flush=True)
    write(output / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--budget-multiple", type=int, default=24)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
