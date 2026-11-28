"""Compare mean and label-conditioned message passing under heterophily."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def network(real, seed):
    if real:
        with np.load(ROOT / "datasets/amazon_ratings.npz", allow_pickle=False) as data:
            x, labels, edges = (
                data[key] for key in ("node_features", "node_labels", "edges")
            )
            partitions = [
                tuple(
                    np.flatnonzero(data[f"{name}_masks"][i])
                    for name in ("train", "val", "test")
                )
                for i in range(3)
            ]
        assert x.shape == (24492, 300) and edges.shape[0] == 93050
        return x, labels.ravel().astype(int), edges, partitions
    rng = np.random.default_rng(seed)
    n, classes = 24000, 5
    labels = rng.integers(classes, size=n)
    pools = [np.flatnonzero(labels == c) for c in range(classes)]
    transitions = np.array([0.08, 0.52, 0.30, 0.07, 0.03])
    source = np.repeat(np.arange(n), 4)
    offsets = rng.choice(classes, size=len(source), p=transitions)
    dst_type = (labels[source] + offsets) % classes
    destination = np.empty(len(source), dtype=int)
    for c, pool in enumerate(pools):
        take = dst_type == c
        destination[take] = rng.choice(pool, size=take.sum(), replace=True)
    x = rng.normal(0, 1.0, (n, 96))
    centers = rng.normal(0, 0.48, (classes, 96))
    x += centers[labels] * rng.uniform(0.12, 0.85, (n, 1))
    partitions = []
    for i in range(3):
        tr_v, test = train_test_split(
            np.arange(n), test_size=0.3, stratify=labels, random_state=seed + i
        )
        train, val = train_test_split(
            tr_v, test_size=2 / 7, stratify=labels[tr_v], random_state=seed + i + 3
        )
        partitions.append((train, val, test))
    return (
        x.astype(np.float32),
        labels,
        np.stack([source, destination], axis=1),
        partitions,
    )


def graph_adjacency(edges, n):
    source, destination = edges.T
    keep = source != destination
    a, b = source[keep], destination[keep]
    both = np.r_[a, b], np.r_[b, a]
    result = coo_matrix(
        (np.ones(len(both[0]), dtype=np.float32), both), shape=(n, n)
    ).tocsr()
    result.data[:] = 1
    return result


def class_posterior(x, labels, train, seed):
    k = len(np.unique(labels))
    logits = np.zeros((len(x), k), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    other = np.setdiff1d(np.arange(len(x)), train)
    estimators = []
    for subset, held in splitter.split(train, labels[train]):
        model = LogisticRegression(max_iter=250, C=0.05, random_state=seed)
        model.fit(x[train[subset]], labels[train[subset]])
        logits[train[held]] = model.decision_function(x[train[held]])
        logits[other] += model.decision_function(x[other]) / splitter.n_splits
        estimators.append((model.coef_, model.intercept_))
    return logits, estimators


def build_channels(x, adjacency, logits):
    p = softmax(logits, axis=1).astype(np.float32)
    degree = adjacency.sum(axis=1).A.ravel()
    mean = (adjacency @ x) / np.maximum(degree[:, None], 1)
    channels = {"features": x, "mean_graph": np.column_stack([x, mean])}
    for mode, weights in (
        ("hard_label_graph", np.eye(p.shape[1], dtype=np.float32)[p.argmax(axis=1)]),
        ("soft_label_graph", p),
    ):
        stats = adjacency @ weights
        density = stats / np.maximum(degree[:, None], 1)
        typed = [
            (adjacency @ (x * weights[:, i : i + 1]))
            / np.maximum(stats[:, i : i + 1], 1e-3)
            for i in range(p.shape[1])
        ]
        channels[mode] = np.column_stack([x, density, *typed])
    return channels


class GraphReadout(nn.Module):
    def __init__(self, width, labels):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(width, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.22)
        )
        self.residual = nn.Sequential(
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.22),
            nn.Linear(192, 192),
        )
        self.head = nn.Linear(192, labels)

    def forward(self, x):
        z = self.input(x)
        return self.head(z + self.residual(z))


def evaluate(y, p):
    return {
        "accuracy": float(accuracy_score(y, p.argmax(1))),
        "macro_f1": float(f1_score(y, p.argmax(1), average="macro")),
        "log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
    }


def train_readout(x, y, partitions, seed, epochs, path):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train, val, test = partitions
    norm = StandardScaler().fit(x[train])
    values = torch.from_numpy(np.clip(norm.transform(x), -8, 8))
    targets = torch.from_numpy(y)
    model = GraphReadout(values.shape[1], len(np.unique(y)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0011, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    history = []
    best = (-1.0, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in np.array_split(rng.permutation(train), max(1, len(train) // 256)):
            loss = criterion(model(values[batch]), targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            valid = evaluate(y[val], model(values[val]).softmax(1).numpy())
        history.append(valid)
        if valid["macro_f1"] > best[0]:
            best = (valid["macro_f1"], epoch)
            torch.save(model.state_dict(), path)
        if epoch - best[1] >= 7:
            break
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    with torch.inference_mode():
        p = model(values[test]).softmax(1).numpy()
    return (
        {
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "selected_epoch": best[1],
            "validation_history": history,
            "test": evaluate(y[test], p),
        },
        p,
        norm,
    )


def run(args):
    torch.set_num_threads(1)
    x, y, edges, splits = network(args.real, args.seed)
    graph = graph_adjacency(edges, len(y))
    output = ROOT / (
        "real-data-projects/wang-amazon-heterophilic-graph"
        if args.real
        else "professor-aligned/wang-label-wise-graph"
    )
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "configuration": vars(args),
        "nodes": len(y),
        "directed_edges_after_symmetrization": graph.nnz,
        "edge_homophily": float(np.mean(y[edges[:, 0]] == y[edges[:, 1]])),
        "splits": {},
    }
    for i, parts in enumerate(splits[: args.splits]):
        train, val, test = parts
        assert len(np.unique(np.concatenate(parts))) == sum(map(len, parts)) == len(y)
        norm = StandardScaler().fit(x[train])
        scaled = np.clip(norm.transform(x), -8, 8)
        reduction = PCA(
            n_components=64, svd_solver="randomized", random_state=args.seed
        ).fit(scaled[train])
        z = reduction.transform(scaled).astype(np.float32)
        p, estimators = class_posterior(z, y, train, args.seed + i)
        coefficients, offsets = zip(*estimators)
        np.savez_compressed(
            output / f"preprocessing-{i}.npz",
            train=train,
            validation=val,
            test=test,
            zero=norm.mean_,
            scale=norm.scale_,
            pca_mean=reduction.mean_,
            pca_components=reduction.components_,
            posterior_coef=np.asarray(coefficients),
            posterior_intercept=np.asarray(offsets),
        )
        readouts = build_channels(z, graph, p)
        metrics = {}
        for name, values in readouts.items():
            scores, predictions, normalization = train_readout(
                values, y, parts, args.seed + i, args.epochs, output / f"{name}-{i}.pt"
            )
            metrics[name] = scores
            np.savez_compressed(
                output / f"{name}-{i}-test.npz",
                labels=y[test],
                predictions=predictions,
                feature_mean=normalization.mean_,
                feature_scale=normalization.scale_,
            )
            print(i, name, scores["test"], flush=True)
        results["splits"][str(i)] = metrics
        (output / "progress.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    results["mean_test"] = {
        name: {
            metric: float(
                np.mean(
                    [fold[name]["test"][metric] for fold in results["splits"].values()]
                )
            )
            for metric in ("accuracy", "macro_f1", "log_loss")
        }
        for name in metrics
    }
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--splits", type=int, choices=(1, 2, 3), default=3)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
