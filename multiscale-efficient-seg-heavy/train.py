import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", ROOT / "data"))
SEEDS = [13, 37, 61]


def save(obj, name):
    folder = ROOT / "checkpoints"
    folder.mkdir(exist_ok=True)
    joblib.dump(obj, folder / (name + ".joblib"), compress=3)


def scores(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def dataset(seed):
    rng = np.random.default_rng(seed)
    u, v = np.mgrid[-1:1:40j, -1:1:40j]
    x = []
    y = []
    for _ in range(480):
        a, b = rng.uniform(-0.5, 0.5, 2)
        mask = ((u - a) / rng.uniform(0.15, 0.55)) ** 2 + (
            (v - b) / rng.uniform(0.14, 0.49)
        ) ** 2 < 1
        stripes = 0.11 * np.sin(u * rng.uniform(5, 12) + v * rng.uniform(2, 7))
        texture = rng.normal(0.40, 0.20, (40, 40, 3)) + stripes[:, :, None]
        x.append(np.clip(texture + mask[:, :, None] * [0.17, 0.08, -0.05], 0, 1))
        y.append(mask)
    return np.asarray(x), np.asarray(y)


import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(2)


class Segmenter(nn.Module):
    def __init__(self, efficient):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(16, 24, 3, padding=1), nn.ReLU()
        )
        self.efficient = efficient
        self.fusion = nn.Sequential(nn.Conv2d(40, 24, 1), nn.ReLU())
        if efficient:
            self.branches = nn.ModuleList(
                nn.Conv2d(24, 24, k, padding=k // 2, groups=24) for k in (3, 5)
            )
            self.mix = nn.Conv2d(48, 24, 1)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(24, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, 24, 1),
                nn.Sigmoid(),
            )
        else:
            self.context = nn.Sequential(
                nn.Conv2d(24, 24, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, 24, 3, padding=1),
            )
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, x):
        skip = self.stem(x)
        z = self.encoder(skip)
        if self.efficient:
            z = z + self.mix(
                torch.cat([branch(z) for branch in self.branches], dim=1)
            ) * self.gate(z)
        else:
            z = z + self.context(z)
        z = F.interpolate(z, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.fusion(torch.cat([z, skip], dim=1)))


def dice(pred, y):
    p = pred >= 0
    intersection = (p * y).sum((1, 2, 3))
    return (
        ((2 * intersection + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1))
        .mean()
        .item()
    )


def run(seed):
    x, y = dataset(seed)
    x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.float32)[:, None]
    train, rest = train_test_split(np.arange(len(y)), test_size=0.36, random_state=seed)
    valid, test = np.array_split(rest, 2)
    results = {}
    for tag in ["dense", "depthwise"]:
        torch.manual_seed(seed)
        model = Segmenter(tag == "depthwise")
        optim = torch.optim.AdamW(model.parameters(), lr=0.003)
        best = -1
        for _ in range(9):
            model.train()
            for idx in torch.randperm(len(train)).split(24):
                ids = train[idx.numpy()]
                p = model(x[ids])
                target = y[ids]
                f = p.sigmoid()
                overlap = (2 * (f * target).sum() + 1) / (f.sum() + target.sum() + 1)
                loss = F.binary_cross_entropy_with_logits(p, target) + 1 - overlap
                optim.zero_grad()
                loss.backward()
                optim.step()
            model.eval()
            with torch.no_grad():
                val = dice(model(x[valid]), y[valid])
            if val > best:
                best = val
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        folder = ROOT / "checkpoints"
        folder.mkdir(exist_ok=True)
        torch.save(state, folder / f"{tag}-{seed}.pt")
        with torch.no_grad():
            value = dice(model(x[test]), y[test])
        results[tag] = {
            "dice": value,
            "parameters": sum(p.numel() for p in model.parameters()),
            "test_images": len(test),
        }
    return results


if __name__ == "__main__":
    runs = {s: run(s) for s in SEEDS}
    first = next(iter(runs.values()))
    means = {
        k: {m: float(np.mean([r[k][m] for r in runs.values()])) for m in v}
        for k, v in first.items()
    }
    (ROOT / "results.json").write_text(
        json.dumps({"seeds": SEEDS, "runs": runs, "means": means}, indent=2)
    )
    print(means)
