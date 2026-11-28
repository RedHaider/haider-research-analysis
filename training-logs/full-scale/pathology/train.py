import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets/medmnist/pathmnist.npz"
BAG = 12


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.Conv2d(width, 2 * width, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.GELU(),
            nn.Conv2d(2 * width, width, 1, bias=False),
            nn.BatchNorm2d(width),
        )

    def forward(self, x):
        return F.gelu(x + self.layers(x))


class BagClassifier(nn.Module):
    def __init__(self, gated):
        super().__init__()
        self.gated = gated
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, 4, stride=4),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 144, 3, stride=2, padding=1),
            nn.BatchNorm2d(144),
            nn.GELU(),
            Residual(144),
            nn.Conv2d(144, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            Residual(256),
            Residual(256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        if gated:
            self.a = nn.Linear(256, 128)
            self.b = nn.Linear(256, 128)
            self.c = nn.Linear(128, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        batch, instances = x.shape[:2]
        features = self.encoder(
            x.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        )
        features = features.reshape(batch, instances, -1)
        if self.gated:
            attention = self.c(
                self.a(features).tanh() * self.b(features).sigmoid()
            ).softmax(dim=1)
            pooled = (features * attention).sum(dim=1)
        else:
            pooled = features.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def synthetic(count, seed):
    rng = np.random.default_rng(seed)
    u, v = np.mgrid[-1:1:28j, -1:1:28j]
    labels = rng.random(count) < 0.11
    result = np.empty((count, 28, 28, 3), np.uint8)
    for i, label in enumerate(labels):
        image = rng.normal(0.54, 0.07, (28, 28, 3))
        for _ in range(int(rng.integers(4, 11))):
            a, b = rng.uniform(-1, 1, 2)
            mask = np.exp(-((u - a) ** 2 + (v - b) ** 2) / rng.uniform(0.004, 0.03))
            image -= mask[:, :, None] * rng.uniform(0.10, 0.27)
        if label:
            a, b = rng.uniform(-0.55, 0.55, 2)
            region = np.exp(-((u - a) ** 2 + (v - b) ** 2) / 0.047)
            image += region[:, :, None] * np.array([0.16, -0.17, 0.09])
        result[i] = np.rint(255 * np.clip(image, 0, 1)).astype(np.uint8)
    return result, labels


def partitions(real):
    if real:
        with np.load(DATA) as archive:
            return {
                key: (archive[key + "_images"], archive[key + "_labels"].ravel() == 8)
                for key in ("train", "val", "test")
            }
    return {
        key: synthetic(count, seed)
        for key, count, seed in (
            ("train", 18000, 170),
            ("val", 3600, 171),
            ("test", 3600, 172),
        )
    }


def make_bags(y, seed):
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.flatnonzero(y))
    neg = rng.permutation(np.flatnonzero(~y))
    # Use distinct tiles within a split. Positive bags contain a small number of
    # positive tiles, while negatives consist entirely of other tissue types.
    count = min(len(y) // BAG, len(neg) // (BAG - 1))
    wanted = min(count // 2, len(pos))
    bags = []
    labels = []
    p = n = 0
    for k in range(count):
        tumor = 0
        if k < wanted:
            tumor = min(1 + (k % 3), len(pos) - p - (wanted - k - 1))
        normal = BAG - tumor
        if n + normal > len(neg):
            break
        ix = np.r_[pos[p : p + tumor], neg[n : n + normal]]
        rng.shuffle(ix)
        bags.append(ix)
        labels.append(tumor > 0)
        p += tumor
        n += normal
    bags = np.array(bags)
    labels = np.asarray(labels, dtype=np.float32)
    order = rng.permutation(len(bags))
    assert len(set(bags.ravel())) == bags.size and len(np.unique(labels)) == 2
    return bags[order], labels[order]


def run(parts, readouts, out, seed, arm, epochs, device):
    torch.manual_seed(seed)
    encoder = BagClassifier(arm == "gated").to(
        device, memory_format=torch.channels_last
    )
    opt = torch.optim.AdamW(encoder.parameters(), lr=0.0014, weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)

    def input_batch(part, ids):
        pixels, bags, _ = readouts[part]
        x = torch.from_numpy(pixels[bags[ids]]).permute(0, 1, 4, 2, 3)
        return x.to(device=device, dtype=torch.float32).div_(127.5).sub_(1)

    def evaluate(part):
        encoder.eval()
        _, bags, y = readouts[part]
        pred = []
        with torch.inference_mode():
            for ids in np.array_split(
                np.arange(len(bags)), max(1, len(bags) // 24 + 1)
            ):
                pred.append(encoder(input_batch(part, ids)).sigmoid().cpu().numpy())
        pred = np.concatenate(pred)
        return {
            "auroc": float(roc_auc_score(y, pred)),
            "average_precision": float(average_precision_score(y, pred)),
        }

    n = len(readouts["train"][2])
    history = []
    best = -1.0
    best_epoch = 0
    path = out / "checkpoints" / f"{arm}-{seed}.pt"
    for epoch in range(epochs):
        start = time.perf_counter()
        encoder.train()
        for ids in np.array_split(rng.permutation(n), max(1, n // 24 + 1)):
            labels = torch.from_numpy(readouts["train"][2][ids]).to(device)
            x = input_batch("train", ids)
            if rng.random() < 0.5:
                x = x.flip(-1)
            loss = F.binary_cross_entropy_with_logits(encoder(x), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 4.0)
            opt.step()
        scheduler.step()
        val = evaluate("val")
        history.append(
            {**val, "epoch": epoch + 1, "epoch_seconds": time.perf_counter() - start}
        )
        print(seed, arm, history[-1], flush=True)
        if val["average_precision"] > best + 1e-5:
            best = val["average_precision"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "state": encoder.state_dict(),
                    "epoch": best_epoch,
                    "history": history,
                },
                path,
            )
        if epoch + 1 - best_epoch >= 3:
            break
    chosen = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(chosen["state"])
    result = {
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "selected_epoch": best_epoch,
        "history": history,
        "test": evaluate("test"),
    }
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--real", action="store_true")
    cli.add_argument("--epochs", type=int, default=10)
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = cli.parse_args()
    if args.epochs < 1:
        cli.error("epochs must be positive")
    torch.set_num_threads(args.threads)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    prefix = "real-data-projects" if args.real else "professor-aligned"
    out = (
        ROOT
        / prefix
        / (
            "amal-pathology-mil-real-data"
            if args.real
            else "amal-pathology-mil-aligned"
        )
    )
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    parts = partitions(args.real)
    readouts = {}
    for i, (name, (x, y)) in enumerate(parts.items()):
        ids, lab = make_bags(y, 417 + i)
        readouts[name] = (x, ids, lab)
    np.savez_compressed(
        out / "checkpoints/splits.npz",
        **{
            f"{k}_{s}": r[j]
            for k, r in readouts.items()
            for j, s in ((1, "indices"), (2, "labels"))
        },
    )
    results = {
        "dataset": "PathMNIST" if args.real else "synthetic tissue tiles",
        "device": device,
        "max_epochs": args.epochs,
        "splits": {
            k: {
                "available_tiles": len(x),
                "used_tiles": int(bags.size),
                "bags": len(bags),
                "positive_bags": int(y.sum()),
            }
            for k, (x, bags, y) in readouts.items()
        },
        "runs": {},
    }
    for seed in [13, 37]:
        runs = {}
        for arm in ("mean", "gated"):
            runs[arm] = run(parts, readouts, out, seed, arm, args.epochs, device)
        results["runs"][seed] = runs
        (out / "progress.json").write_text(json.dumps(results, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
