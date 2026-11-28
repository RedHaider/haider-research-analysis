"""Train frozen nuisance projection and expression prediction under cohort shift."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]


def load_metabric():
    dataset = ROOT / "datasets"
    metadata = pd.read_csv(dataset / "METABRIC_metadata.tsv", sep="\t", dtype=str)
    matrix = pd.read_csv(dataset / "METABRIC.tsv.gz", sep="\t", low_memory=False)
    joined = metadata.loc[metadata["Sample_ID"].isin(matrix.columns)].copy()
    joined["label"] = (
        joined["ER_IHC"]
        .str.strip()
        .str.upper()
        .map({"POSITIVE": 1, "POSITVE": 1, "NEGATIVE": 0})
    )
    joined["cohort"] = pd.to_numeric(joined["COHORT"], errors="coerce")
    joined = joined.dropna(subset=["label", "cohort"])
    if joined["Sample_ID"].duplicated().any():
        raise ValueError("METABRIC sample IDs must be unique")
    genes = matrix["HGNC_Symbol"].fillna("")
    matrix = matrix.loc[
        genes.str.upper() != "ESR1", joined["Sample_ID"].tolist()
    ].apply(pd.to_numeric, errors="coerce")
    return (
        matrix.to_numpy(dtype=np.float32).T,
        joined["label"].to_numpy(dtype=int),
        joined["cohort"].to_numpy(dtype=int),
        joined["Sample_ID"].to_numpy(),
    )


def simulate():
    rng = np.random.default_rng(2027)
    n_cohorts, per_cohort, genes = 5, 480, 4096
    cohorts = np.repeat(np.arange(n_cohorts), per_cohort)
    label_rates = np.array([0.2, 0.8, 0.4, 0.7, 0.5])
    labels = rng.binomial(1, label_rates[cohorts])
    nuisance = rng.normal(0, 0.4, (n_cohorts * per_cohort, 8))
    nuisance[:, 0] += np.array([-2, 2, -1, 1, 0])[cohorts]
    nuisance[:, 1] += np.array([0, 0.5, -1, 0.5, 2])[cohorts]
    nuisance[:, 2] += (2 * labels - 1) * np.where(cohorts == 4, -1.35, 1.35)
    nuisance_loadings = rng.normal(0, 0.25, (8, genes))
    biological_loadings = np.zeros(genes)
    biological_loadings[:120] = rng.normal(0, 0.2, 120)
    x = nuisance @ nuisance_loadings + (2 * labels[:, None] - 1) * biological_loadings
    x += rng.normal(0, 1, x.shape)
    return x.astype(np.float32), labels, cohorts, np.arange(len(labels)).astype(str)


class FrozenProjection:
    """Estimate nuisance directions from training residuals and freeze them."""

    def fit(self, x, labels, dimensions):
        design = np.stack([np.ones(len(labels)), labels], axis=1)
        fit = np.linalg.lstsq(design, x, rcond=None)[0]
        residuals = (x - design @ fit).astype(np.float32)
        self.axes = (
            PCA(n_components=dimensions, svd_solver="randomized", random_state=27)
            .fit(residuals)
            .components_
        )
        return self

    def transform(self, x):
        return x - (x @ self.axes.T) @ self.axes


class FrozenControlCorrection:
    """Use a frozen control-feature encoder and a training-only outcome-adjusted regression."""

    def fit(self, x, labels, groups, dimensions, known_controls):
        if known_controls is not None:
            self.controls = np.flatnonzero(known_controls)
        else:
            associations = []
            for cohort in np.unique(groups):
                values = x[groups == cohort]
                outcomes = labels[groups == cohort].astype(float)
                if len(np.unique(outcomes)) < 2:
                    continue
                centered = values - values.mean(axis=0)
                centered_y = outcomes - outcomes.mean()
                denom = np.sqrt((centered**2).sum(axis=0) * (centered_y**2).sum())
                associations.append(
                    np.abs(centered_y @ centered) / np.maximum(denom, 1e-8)
                )
            max_association = np.max(associations, axis=0)
            self.controls = np.argsort(max_association, kind="stable")[
                : x.shape[1] // 2
            ]
        pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=27)
        latent = pca.fit_transform(x[:, self.controls])
        self.center, self.axes = pca.mean_, pca.components_
        design = np.column_stack([np.ones(len(x)), labels, latent])
        self.effects = np.linalg.lstsq(design, x, rcond=None)[0][2:].astype(np.float32)
        return self

    def transform(self, x):
        latent = (x[:, self.controls] - self.center) @ self.axes.T
        return x - latent @ self.effects


class ResidualPredictor(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(features, 320), nn.LayerNorm(320), nn.GELU(), nn.Dropout(0.2)
        )
        self.residual = nn.Sequential(
            nn.Linear(320, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(320, 320),
        )
        self.head = nn.Linear(320, 1)

    def forward(self, x):
        h = self.project(x)
        return self.head(F.gelu(h + self.residual(h))).squeeze(-1)


def scores(y, pred):
    return {
        "auroc": float(roc_auc_score(y, pred)),
        "ap": float(average_precision_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
    }


def standardize(x, indexes):
    train = x[indexes]
    present = np.isfinite(train).mean(axis=0) >= 0.99
    train = train[:, present]
    train[~np.isfinite(train)] = np.nan
    var = np.nanvar(train, axis=0)
    selected = np.flatnonzero(var > 1e-8)
    selected = selected[np.argsort(-var[selected], kind="stable")[:4096]]
    columns = np.flatnonzero(present)[selected]
    mean, scale = np.nanmean(train[:, selected], axis=0), np.sqrt(var[selected])
    values = (x[:, columns] - mean) / scale
    values = np.clip(np.where(np.isfinite(values), values, 0), -8, 8)
    return values.astype(np.float32), (columns, mean, scale)


def train_mlp(x, y, train, val, test, seed, epochs, output):
    torch.manual_seed(seed)
    values, labels = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    model = ResidualPredictor(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    rng = np.random.default_rng(seed)
    history, best = [], (-1.0, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in np.array_split(rng.permutation(train), max(1, len(train) // 64)):
            loss = F.binary_cross_entropy_with_logits(
                model(values[indices]), labels[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            validation = scores(y[val], model(values[val]).sigmoid().numpy())
        history.append(validation)
        if validation["ap"] > best[0]:
            best = (validation["ap"], epoch)
            torch.save(model.state_dict(), output)
        if epoch - best[1] >= 7:
            break
    model.load_state_dict(torch.load(output, weights_only=True))
    model.eval()
    with torch.inference_mode():
        pred = model(values[test]).sigmoid().numpy()
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best[1],
        "history": history,
        "test": scores(y[test], pred),
    }, pred


def write(path, obj):
    part = path.with_suffix(".tmp")
    part.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


def main(args):
    torch.set_num_threads(args.threads)
    output = ROOT / (
        "real-data-projects/leek-metabric-frozen-correction"
        if args.real
        else "professor-aligned/leek-frozen-surrogate-shift"
    )
    output.mkdir(parents=True, exist_ok=True)
    x, y, cohorts, ids = load_metabric() if args.real else simulate()
    assert len(np.unique(ids)) == len(ids)
    results = {
        "samples": len(x),
        "raw_features": x.shape[1],
        "configuration": vars(args),
        "cohorts": {},
    }
    splits = {}
    eligible = [
        cohort
        for cohort in np.unique(cohorts)
        if np.bincount(y[cohorts == cohort], minlength=2).min() >= 8
    ]
    if len(eligible) < 3:
        raise ValueError(f"Expected at least three evaluable cohorts, found {eligible}")
    for cohort in eligible:
        test = np.flatnonzero(cohorts == cohort)
        train, val = train_test_split(
            np.flatnonzero(cohorts != cohort),
            test_size=0.2,
            stratify=y[cohorts != cohort],
            random_state=19,
        )
        splits[str(cohort)] = {
            name: ids[indices].tolist()
            for name, indices in (("train", train), ("validation", val), ("test", test))
        }
        write(output / "splits.json", splits)
        z, (columns, mean, scale) = standardize(x, train)
        projection = FrozenProjection().fit(z[train], y[train], args.dimensions)
        corrected = projection.transform(z)
        controls = FrozenControlCorrection().fit(
            z[train],
            y[train],
            cohorts[train],
            args.dimensions,
            None if args.real else columns >= 120,
        )
        control_corrected = controls.transform(z)
        np.savez_compressed(
            output / f"preprocessing-{cohort}.npz",
            columns=columns,
            mean=mean,
            scale=scale,
            nuisance_axes=projection.axes,
            controls=controls.controls,
            control_center=controls.center,
            control_axes=controls.axes,
            control_effects=controls.effects,
        )
        models, preds = {}, {}
        linear = LogisticRegression(C=0.05, max_iter=1000, random_state=19).fit(
            z[train], y[train]
        )
        preds["logistic"] = linear.predict_proba(z[test])[:, 1]
        models["logistic"] = {"test": scores(y[test], preds["logistic"])}
        np.savez(
            output / f"logistic-{cohort}.npz",
            weights=linear.coef_,
            bias=linear.intercept_,
        )
        for name, values in (
            ("raw_mlp", z),
            ("frozen_corrected_mlp", corrected),
            ("frozen_controls_mlp", control_corrected),
        ):
            models[name], preds[name] = train_mlp(
                values,
                y,
                train,
                val,
                test,
                19,
                args.epochs,
                output / f"{name}-{cohort}.pt",
            )
        np.savez_compressed(
            output / f"predictions-{cohort}.npz",
            ids=ids[test].astype(str),
            labels=y[test],
            **preds,
        )
        results["cohorts"][str(cohort)] = {
            "test_size": len(test),
            "test_prevalence": float(y[test].mean()),
            "models": models,
        }
        write(output / "progress.json", results)
        print(
            cohort,
            {name: metrics["test"] for name, metrics in models.items()},
            flush=True,
        )
    results["macro_test"] = {
        name: {
            metric: float(
                np.mean(
                    [
                        r["models"][name]["test"][metric]
                        for r in results["cohorts"].values()
                    ]
                )
            )
            for metric in ("auroc", "ap", "brier")
        }
        for name in models
    }
    write(output / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dimensions", type=int, default=8)
    main(parser.parse_args())
