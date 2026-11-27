import argparse
import json
from pathlib import Path

import numpy as np
import torch
from data import load_data
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
torch.set_num_threads(1)
SEEDS = (13, 37, 61)


class Residual(nn.Module):
    def __init__(self, width, conv, norm):
        super().__init__()
        self.layers = nn.Sequential(
            conv(width, width, 3, padding=1),
            norm(width),
            nn.GELU(),
            conv(width, width, 3, padding=1),
            norm(width),
        )

    def forward(self, x):
        return F.gelu(x + self.layers(x))


class Model(nn.Module):
    def __init__(self, channels, dim, outputs, large, dense):
        super().__init__()
        conv = (nn.Conv1d, nn.Conv2d)[dim - 1]
        norm = (nn.BatchNorm1d, nn.BatchNorm2d)[dim - 1]
        avg = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)[dim - 1]
        low, high = (24, 48) if large else (12, 20)
        self.stem = nn.Sequential(
            conv(channels, low, 3, padding=1), norm(low), nn.GELU()
        )
        self.context = nn.Sequential(
            conv(low, high, 3, stride=2, padding=1),
            norm(high),
            nn.GELU(),
            *(
                [Residual(high, conv, norm), Residual(high, conv, norm)]
                if large
                else []
            ),
        )
        self.dense = dense
        if dense:
            self.head = nn.Sequential(
                conv(low + high, low, 3, padding=1), nn.GELU(), conv(low, outputs, 1)
            )
        else:
            self.head = nn.Sequential(
                avg(1),
                nn.Flatten(),
                nn.Linear(high, high * 2),
                nn.GELU(),
                nn.Dropout(0.12),
                nn.Linear(high * 2, outputs),
            )

    def forward(self, x):
        skip = self.stem(x)
        z = self.context(skip)
        if self.dense:
            z = F.interpolate(
                z, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            return self.head(torch.cat([skip, z], 1))
        return self.head(z)


def split(data, seed):
    g = data["groups"]
    official = data.get("test")
    if official is None:
        pool, te = next(
            GroupShuffleSplit(n_splits=1, test_size=0.23, random_state=seed).split(
                g, groups=g
            )
        )
    else:
        pool, te = np.flatnonzero(~official), np.flatnonzero(official)
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=seed + 9).split(
            g[pool], groups=g[pool]
        )
    )
    tr, va = pool[a], pool[b]
    assert (
        not set(g[tr]) & set(g[te])
        and not set(g[va]) & set(g[te])
        and not set(g[tr]) & set(g[va])
    )
    return tr, va, te


def evaluate(y, p, kind):
    if kind == "classification":
        proba = torch.from_numpy(p).softmax(1).numpy()
        out = {"accuracy": accuracy_score(y, proba.argmax(1))}
        if p.shape[1] == 2:
            out["auroc"] = roc_auc_score(y, proba[:, 1])
            out["brier"] = brier_score_loss(y, proba[:, 1])
        return out
    if kind == "coordinates":
        return {"mean_error_px": float(np.linalg.norm(y - p, axis=1).mean() * 20)}
    if kind == "mask":
        pred = p >= 0
        return {
            "dice": float(
                np.mean(
                    (2 * (pred * y).sum((1, 2, 3)) + 1)
                    / (pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
                )
            )
        }
    mse = np.mean((np.clip(p, 0, 1) - y) ** 2)
    return {
        "psnr_db": float(-10 * np.log10(max(1e-12, mse))),
        "mae": float(np.abs(y - np.clip(p, 0, 1)).mean()),
    }


def run(data, seed, folder, idx):
    x = np.asarray(data["x"], np.float32)
    y = np.asarray(data["y"])
    kind = data["kind"]
    outputs = data.get("outputs", 1)
    tr, va, te = split(data, seed)
    rng = np.random.default_rng(seed)
    if len(tr) > 2200:
        tr = np.sort(rng.choice(tr, 2200, replace=False))
    axes = (0, *range(2, x.ndim))
    shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
    mean = x[tr].mean(axis=axes).reshape(shape)
    std = x[tr].std(axis=axes).reshape(shape) + 1e-5

    def tensor(a):
        return torch.as_tensor(np.clip((a - mean) / std, -10, 10), dtype=torch.float32)

    x0 = tensor(x)
    yt = torch.as_tensor(
        y, dtype=torch.long if kind == "classification" else torch.float32
    )
    xp = tensor(data["stress"]) if "stress" in data else None
    out = {}
    for large in (False, True):
        name = "residual" if large else "compact"
        torch.manual_seed(seed)
        model = Model(x.shape[1], x.ndim - 2, outputs, large, kind in ("mask", "image"))
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
        best = float("inf")

        def loss(p, t):
            if kind == "classification":
                if outputs == 2 and idx in (0, 4):
                    w = torch.bincount(yt[tr], minlength=2).float().clamp(min=1)
                    return F.cross_entropy(p, t, weight=w.sum() / (2 * w))
                return F.cross_entropy(p, t)
            if kind == "mask":
                q = p.sigmoid()
                dice = (2 * (q * t).sum() + 1) / (q.sum() + t.sum() + 1)
                weight = ((1 - t.mean()) / t.mean().clamp(min=0.02)).clamp(max=10)
                return (
                    F.binary_cross_entropy_with_logits(p, t, pos_weight=weight)
                    + 1
                    - dice
                )
            return F.smooth_l1_loss(p, t, beta=0.08)

        def predict(arr, ids):
            model.eval()
            with torch.inference_mode():
                return torch.cat(
                    [
                        model(arr[j])
                        for j in np.array_split(ids, max(1, len(ids) // 95 + 1))
                    ]
                )

        for epoch in range(10 if kind == "mask" else 6):
            model.train()
            for ids in torch.randperm(len(tr)).split(8 if kind == "mask" else 64):
                rows = tr[ids.numpy()]
                batch = x0[rows]
                if idx == 7 and large:
                    batch = batch.clone()
                    batch[::3, 3:] = tensor(np.zeros_like(x[rows[:1]]))[0, 3:]
                loss_value = loss(model(batch), yt[rows])
                opt.zero_grad()
                loss_value.backward()
                opt.step()
            validation = loss(predict(x0, va), yt[va]).item()
            if validation < best:
                best = validation
                best_epoch = epoch + 1
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        torch.save(
            {"weights": state, "epoch": best_epoch}, folder / f"{name}-{seed}.pt"
        )
        pred = predict(x0, te).numpy()
        out[name] = {
            **evaluate(y[te], pred, kind),
            "parameters": sum(p.numel() for p in model.parameters()),
            "epoch": best_epoch,
        }
        if xp is not None:
            out[name + "_stress"] = evaluate(y[te], predict(xp, te).numpy(), kind)
        if idx == 9:
            vl = predict(x0, va)
            temps = np.geomspace(0.45, 4, 51)
            t = min(temps, key=lambda a: F.cross_entropy(vl / float(a), yt[va]).item())
            proba = torch.from_numpy(pred / float(t)).softmax(1).numpy()
            order = np.argsort(-proba.max(1))[: int(len(te) * 0.8)]
            out[name + "_calibrated"] = {
                **evaluate(y[te], pred / float(t), kind),
                "temperature": float(t),
                "error_at_80pct": float(
                    np.mean(proba[order].argmax(1) != y[te][order])
                ),
            }
    if "baseline" in data:
        out["input_baseline"] = evaluate(y[te], data["baseline"][te], kind)
    np.savez(
        folder / f"split-{seed}.npz",
        train=tr,
        validation=va,
        test=te,
        mean=mean,
        scale=std,
    )
    return {
        "sizes": [len(tr), len(va), len(te)],
        "groups": [len(set(data["groups"][k])) for k in (tr, va, te)],
        "metrics": out,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("index", type=int)
    cli.add_argument("--real", action="store_true")
    args = cli.parse_args()
    records = json.loads((ROOT / "training-logs/training-batch-50.json").read_text())
    rec = next(r for r in records if r["i"] == args.index and r["real"] == args.real)
    output = ROOT / rec["folder"]
    folder = output / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    data = load_data(DATA, args.index, args.real)
    runs = {seed: run(data, seed, folder, args.index) for seed in SEEDS}
    means = {
        arm: {
            k: float(np.mean([r["metrics"][arm][k] for r in runs.values()]))
            for k in row
        }
        for arm, row in runs[13]["metrics"].items()
    }
    (output / "results.json").write_text(
        json.dumps(
            {"record": rec, "samples": len(data["y"]), "runs": runs, "means": means},
            indent=2,
        )
    )
    print(json.dumps(means), flush=True)
