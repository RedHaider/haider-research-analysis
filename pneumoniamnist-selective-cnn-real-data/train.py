"""Selective chest X-ray classification under a paired image corruption."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
SEEDS = [17, 29, 43]


def corrupt(x, g):
    # Global exposure, blur and additive readout noise, identical at test for both arms.
    gain = 0.55 + 0.60 * torch.rand((len(x), 1, 1, 1), generator=g)
    noise = torch.randn(x.shape, generator=g) * 0.075
    return (
        gain * F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1) + noise
    ).clamp(0, 1)


def network():
    layers = []
    for a, b in [(1, 16), (16, 32), (32, 48)]:
        layers.extend(
            [
                nn.Conv2d(a, b, 3, padding=1, bias=False),
                nn.BatchNorm2d(b),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        )
    return nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(48 * 3 * 3, 64),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(64, 1),
        nn.Flatten(start_dim=0),
    )


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(part).sigmoid() for part in x.split(128)]).numpy()


def metrics(p, y):
    error = (p >= 0.5) != y
    order = np.argsort(-np.abs(p - 0.5), kind="stable")
    risks = np.cumsum(error[order]) / np.arange(1, len(y) + 1)
    return {
        "auroc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "aurc": risks.mean(),
        **{f"risk_at_{c}": risks[int(len(y) * c / 100) - 1] for c in (50, 80, 100)},
    }


def run(data, seed):
    train, y = data["train"]
    val, vy = data["val"]
    test, ty = data["test"]
    test_shift = corrupt(test, torch.Generator().manual_seed(9001))
    output = {}
    for method in ["standard", "augmented"]:
        torch.manual_seed(seed)
        model = network()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        best = float("inf")
        trace = []
        g = torch.Generator().manual_seed(seed + 1000)
        for epoch in range(12):
            model.train()
            for ids in torch.randperm(len(y)).split(96):
                x = train[ids].clone()
                if method == "augmented":
                    x[::2] = corrupt(x[::2], g)
                loss = F.binary_cross_entropy_with_logits(model(x), y[ids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            score = brier_score_loss(vy, predict(model, val))
            trace.append(score)
            if score < best:
                best = score
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        checkpoint = ROOT / "checkpoints"
        checkpoint.mkdir(exist_ok=True)
        torch.save(state, checkpoint / f"{method}-{seed}.pt")
        output[method] = {
            "clean": metrics(predict(model, test), ty.numpy()),
            "shifted": metrics(predict(model, test_shift), ty.numpy()),
            "validation_brier": trace,
        }
        print(seed, method, output[method]["shifted"], flush=True)
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/pneumoniamnist.npz",
    )
    path = cli.parse_args().data
    with np.load(path) as raw:
        data = {
            split: (
                torch.tensor(raw[split + "_images"][:, None], dtype=torch.float32)
                / 255,
                torch.tensor(raw[split + "_labels"].flatten(), dtype=torch.float32),
            )
            for split in ("train", "val", "test")
        }
    assert data["train"][0].shape[2:] == (28, 28)
    runs = {s: run(data, s) for s in SEEDS}
    means = {
        m: {
            split: {
                k: np.mean([r[m][split][k] for r in runs.values()])
                for k in runs[17][m][split]
            }
            for split in ("clean", "shifted")
        }
        for m in runs[17]
    }
    (ROOT / "results.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "split_sizes": {k: len(v[1]) for k, v in data.items()},
                "runs": runs,
                "means": means,
            },
            indent=2,
        )
    )
