"""Estimate replicate agreement from ranked measurements and evaluate on held-out groups."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtri
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))


def load_track(file):
    columns = [
        "chromosome",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signal",
        "pvalue",
        "qvalue",
        "summit",
    ]
    table = pd.read_csv(file, sep="\t", compression="gzip", comment="#", names=columns)
    table = table.loc[
        table["chromosome"].isin(AUTOSOMES) & (table["signal"] > 0)
    ].copy()
    offset = table["summit"].where(
        table["summit"] >= 0, (table["end"] - table["start"]) // 2
    )
    table["position"] = table["start"] + offset
    assert len(table) > 10000
    return table


def real_peaks():
    directory = ROOT / "datasets/encode-ctcf"
    source = [load_track(directory / f"uw-replicate{i}.narrowPeak.gz") for i in (1, 2)]
    reference = load_track(directory / "stanford-validation.narrowPeak.gz")
    records = []
    for table in source:
        binned = (
            table.assign(bin=table.position // 500)
            .groupby(["chromosome", "bin"])["signal"]
            .max()
        )
        records.append(binned)
    peaks = pd.concat(records, axis=1).fillna(0)
    peaks.columns = ["replicate1", "replicate2"]
    peaks = peaks.reset_index()
    groups = peaks["chromosome"].to_numpy()
    positions = peaks["bin"].to_numpy() * 500 + 250
    hits = np.zeros(len(peaks), dtype=bool)
    for chrom in AUTOSOMES:
        source_pos = positions[groups == chrom]
        truth = np.sort(
            reference.loc[reference.chromosome == chrom, "position"].to_numpy()
        )
        if not len(truth):
            continue
        right = np.searchsorted(truth, source_pos)
        neighbors = np.column_stack(
            [truth[np.maximum(right - 1, 0)], truth[np.minimum(right, len(truth) - 1)]]
        )
        hits[groups == chrom] = (
            np.abs(neighbors - source_pos[:, None]).min(axis=1) <= 350
        )
    return peaks[["replicate1", "replicate2"]].to_numpy(), hits, groups, positions


def simulate(seed):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.asarray(AUTOSOMES), 5400)
    n = len(groups)
    target = rng.random(n) < 0.27
    batch = np.repeat(rng.normal(0, 0.25, len(AUTOSOMES)), 5400)
    shared = rng.normal(size=n) * target
    noise = rng.normal(size=(n, 2))
    outlier = (~target) & (rng.random(n) < 0.32)
    direction = rng.integers(0, 2, n)
    noise[np.flatnonzero(outlier), direction[outlier]] += rng.uniform(
        2.0, 4.5, outlier.sum()
    )
    latent = noise + (shared + 2.20 * target + batch)[:, None]
    scores = np.exp(latent * np.array([0.9, 1.25]) + np.array([1.0, -0.2]))
    return scores, target, groups, np.tile(np.arange(5400), len(AUTOSOMES))


class Margins:
    def fit(self, x):
        self.sorted = np.sort(x, axis=0)
        return self

    def uniform(self, x):
        n = len(self.sorted)
        ranks = []
        for column in range(x.shape[1]):
            reference = self.sorted[:, column]
            left = np.searchsorted(reference, x[:, column], side="left")
            right = np.searchsorted(reference, x[:, column], side="right")
            ranks.append((0.5 * (left + right) + 0.5) / (n + 1))
        return np.column_stack(ranks)


class ReproMixture:
    def __init__(self, exclusion_mean, null_variance):
        self.null_mean = np.full(2, exclusion_mean)
        self.null_variance = null_variance
        self.mean = np.array([0.8, 0.8])
        self.covariance = np.array([[0.7, 0.35], [0.35, 0.7]])
        self.prior = 0.33

    @staticmethod
    def density(z, mean, covariance):
        centered = z - mean
        inv = np.linalg.inv(covariance)
        square = np.einsum("ij,jk,ik->i", centered, inv, centered)
        return -0.5 * (
            square + np.linalg.slogdet(covariance)[1] + 2 * np.log(2 * np.pi)
        )

    def joint(self, z):
        null = self.density(
            z, self.null_mean, np.eye(2) * self.null_variance
        ) + np.log1p(-self.prior)
        signal = self.density(z, self.mean, self.covariance) + np.log(self.prior)
        return np.column_stack([null, signal])

    def predict(self, z):
        logp = self.joint(z)
        return np.exp(logp[:, 1] - logsumexp(logp, axis=1))

    def fit(self, z):
        self.history = []
        for _ in range(150):
            weights = self.predict(z)
            self.prior = float(np.clip(weights.mean(), 0.01, 0.95))
            self.mean = np.maximum(0.1, np.average(z, weights=weights, axis=0))
            centered = z - self.mean
            cov = (centered.T * weights) @ centered / weights.sum()
            variance = np.maximum(0.05, np.diag(cov))
            rho = np.clip(cov[0, 1] / np.sqrt(variance.prod()), 0.05, 0.95)
            off_diag = rho * np.sqrt(variance.prod())
            self.covariance = np.array(
                [[variance[0], off_diag], [off_diag, variance[1]]]
            )
            self.history.append(float(logsumexp(self.joint(z), axis=1).mean()))
            if (
                len(self.history) > 3
                and abs(self.history[-1] - self.history[-2]) < 1e-7
            ):
                break
        return self


def score(target, predictions):
    top = np.argsort(-predictions, kind="stable")[: max(1, int(0.1 * len(target)))]
    return {
        "average_precision": float(average_precision_score(target, predictions)),
        "auroc": float(roc_auc_score(target, predictions)),
        "precision_at_top_10_percent": float(target[top].mean()),
    }


def evaluate(x, target, groups, positions, output):
    records, summary = {}, {}
    for fold in range(4):
        test = np.isin(groups, AUTOSOMES[fold::4])
        train = ~test
        transform = Margins().fit(x[train])
        z_full = transform.uniform(x)
        z = ndtri(z_full)
        baseline = ReproMixture(0.0, 1.0).fit(z[train])
        # Estimate an empirical background from the lower-scoring half of each assay.
        pooled = z[train].ravel()
        low = pooled[pooled < np.median(pooled)]
        adaptive = ReproMixture(
            float(low.mean() * 0.5), float(np.clip(low.var() * 2, 0.35, 1.50))
        ).fit(z[train])
        predictions = {
            "minimum_percentile": z_full[test].min(axis=1),
            "geometric_mean_percentile": np.sqrt(z_full[test].prod(axis=1)),
            "concordance_mixture": baseline.predict(z[test]),
            "empirical_null_mixture": adaptive.predict(z[test]),
        }
        records[str(fold)] = {
            "test_chromosomes": list(AUTOSOMES[fold::4]),
            "test_count": int(test.sum()),
            "reference_rate": float(target[test].mean()),
            "metrics": {
                name: score(target[test], values)
                for name, values in predictions.items()
            },
        }
        np.savez_compressed(
            output / f"held-out-{fold}.npz",
            group=groups[test].astype(str),
            position=positions[test],
            expected=target[test],
            **predictions,
        )
        for name, model in (("concordance", baseline), ("empirical_null", adaptive)):
            np.savez_compressed(
                output / f"model-{fold}-{name}.npz",
                margin=transform.sorted,
                mean=model.mean,
                covariance=model.covariance,
                prior=model.prior,
                null_mean=model.null_mean,
                null_variance=model.null_variance,
                history=model.history,
            )
        print(fold, records[str(fold)]["metrics"], flush=True)
    for name in predictions:
        summary[name] = {
            metric: float(
                np.mean(
                    [record["metrics"][name][metric] for record in records.values()]
                )
            )
            for metric in next(iter(records.values()))["metrics"][name]
        }
    return records, summary


def main(args):
    x, target, groups, positions = real_peaks() if args.real else simulate(args.seed)
    output = ROOT / (
        "real-data-projects/qunhua-li-encode-reproducibility"
        if args.real
        else "professor-aligned/qunhua-li-replicate-mixture"
    )
    output.mkdir(parents=True, exist_ok=True)
    records, summary = evaluate(x, target, groups, positions, output)
    report = {
        "configuration": vars(args),
        "total_candidates": len(x),
        "folds": records,
        "mean_test": summary,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=32)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
