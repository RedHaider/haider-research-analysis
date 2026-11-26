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


def phantoms(seed):
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


def dataset(seed):
    x, y = phantoms(seed)
    tr, hold = train_test_split(np.arange(len(y)), test_size=0.36, random_state=seed)
    va, te = np.array_split(hold, 2)
    return x, y, tr, va, te


import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, 12, 3, padding=1), nn.ReLU())
        self.context = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(12, 20, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.ReLU(),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1)
        )
        self.classifier = nn.Linear(20, 1)

    def forward(self, x):
        low = self.stem(x)
        z = self.context(low)
        up = F.interpolate(z, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.mask(torch.cat((low, up), 1)), self.classifier(
            z.mean((2, 3))
        ).flatten()


def accuracy(logits, y):
    mask = logits >= 0
    overlap = (mask * y).sum((1, 2, 3))
    return (
        ((2 * overlap + 1) / (mask.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean().item()
    )


def run(seed):
    images, masks, tr, va, te = dataset(seed)
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    masks = torch.tensor(masks, dtype=torch.float32)[:, None]
    areas = masks.mean((1, 2, 3))
    threshold = areas[tr].median()
    category = (areas > threshold).float()
    schemes = [("single_task", images, False), ("joint", images, True)]
    out = {}
    for name, features, aux in schemes:
        torch.manual_seed(seed)
        model = Net(features.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=0.003)
        best = -1
        for _ in range(10):
            model.train()
            for batch in torch.randperm(len(tr)).split(16):
                ids = tr[batch.numpy()]
                p, c = model(features[ids])
                y = masks[ids]
                soft = p.sigmoid()
                overlap = (2 * (soft * y).sum() + 1) / (soft.sum() + y.sum() + 1)
                weight = (1 - y.mean()) / y.mean().clamp(min=0.02)
                loss = (
                    F.binary_cross_entropy_with_logits(
                        p, y, pos_weight=weight.clamp(max=12)
                    )
                    + 1
                    - overlap
                )
                if aux:
                    loss += 0.45 * F.binary_cross_entropy_with_logits(c, category[ids])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = accuracy(model(features[va])[0], masks[va])
            if val > best:
                best = val
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        folder = ROOT / "checkpoints"
        folder.mkdir(exist_ok=True)
        torch.save(state, folder / f"{name}-{seed}.pt")
        with torch.no_grad():
            pred, logits = model(features[te])
            result = {"dice": accuracy(pred, masks[te]), "test_images": len(te)}
            if aux:
                result["size_auroc"] = roc_auc_score(category[te], logits)
        out[name] = result
    return out


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
