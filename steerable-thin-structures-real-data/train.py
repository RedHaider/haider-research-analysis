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


from PIL import Image


def dataset(seed):
    images = []
    masks = []
    for split, ids in [("train", range(21, 41)), ("val", range(1, 21))]:
        for i in ids:
            root = DATA / "drive" / split
            with Image.open(root / "input" / f"{i:02}.tif") as im:
                channel = im.convert("RGB").getchannel("G")
                images.append(np.asarray(channel.resize((144, 144))) / 255)
            with Image.open(
                root / "label" / f"{i:02}{'_manual1' if split == 'val' else ''}.png"
            ) as im:
                arr = np.asarray(
                    im.convert("L").resize((144, 144), Image.Resampling.NEAREST)
                )
                masks.append(arr > 0 if arr.max() == 1 else arr > 127)
    assert all(0.01 < a.mean() < 0.3 for a in masks)
    tr, va = train_test_split(np.arange(20), test_size=0.2, random_state=seed)
    return np.asarray(images)[..., None], np.asarray(masks), tr, va, np.arange(20, 40)


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
        for _ in range(22):
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
