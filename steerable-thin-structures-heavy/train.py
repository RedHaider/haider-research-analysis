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
    import cv2

    rng = np.random.default_rng(seed)
    images = []
    masks = []
    for _ in range(400):
        mask = np.zeros((56, 56), dtype=np.uint8)
        x = np.linspace(3, 53, 48)
        y = 28 + rng.uniform(4, 19) * np.sin(x / rng.uniform(5, 11) + rng.uniform(0, 4))
        xy = np.stack((x, y), 1).astype(np.int32)
        cv2.polylines(mask, [xy], False, 1, int(rng.integers(1, 3)))
        mask = np.rot90(mask, int(rng.integers(0, 4))).copy()
        noise = rng.normal(0, 0.15, mask.shape)
        distractor = cv2.resize(rng.normal(0, 0.12, (7, 7)), (56, 56))
        images.append(
            np.clip(0.5 + noise + distractor - rng.uniform(0.1, 0.30) * mask, 0, 1)
        )
        masks.append(mask)
    tr, hold = train_test_split(
        np.arange(len(masks)), test_size=0.36, random_state=seed
    )
    va, te = np.array_split(hold, 2)
    return np.asarray(images)[..., None], np.asarray(masks), tr, va, te


from scipy.ndimage import gaussian_filter


def bank(images):
    x = images[:, 0].numpy().astype("float32")
    xx = gaussian_filter(x, (0, 1.15, 1.15), order=(0, 0, 2))
    xy = gaussian_filter(x, (0, 1.15, 1.15), order=(0, 1, 1))
    yy = gaussian_filter(x, (0, 1.15, 1.15), order=(0, 2, 0))
    angles = np.arange(6) * np.pi / 6
    responses = [
        (np.cos(a) ** 2 * xx + 2 * np.cos(a) * np.sin(a) * xy + np.sin(a) ** 2 * yy)
        for a in angles
    ]
    return torch.cat((images, torch.from_numpy(np.stack(responses, axis=1).astype(np.float32) * 8)), 1)


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
    schemes = [("intensity", images, False), ("oriented", bank(images), False)]
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
            result["mask_prevalence"] = masks[te].mean().item()
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
