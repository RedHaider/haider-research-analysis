import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from data import load_task
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
SEEDS = (13, 37)


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


class PointResidual(nn.Module):
    def __init__(self, width, dim):
        super().__init__()
        conv = (nn.Conv1d, nn.Conv2d)[dim - 1]
        norm = (nn.BatchNorm1d, nn.BatchNorm2d)[dim - 1]
        self.spatial = conv(width, width, 3, padding=1, groups=width, bias=False)
        self.point = nn.Sequential(
            conv(width, width * 2, 1, bias=False),
            norm(width * 2),
            nn.GELU(),
            conv(width * 2, width, 1, bias=False),
            norm(width),
        )

    def forward(self, x):
        return F.gelu(x + self.point(self.spatial(x)))


class DeepNetwork(nn.Module):
    def __init__(self, shape, outputs, dense, heavy):
        super().__init__()
        dim = len(shape) - 1
        self.dim = dim
        self.dense = dense
        if dim == 0:
            h1, h2 = (768, 512) if heavy else (64, 32)
            self.mlp = nn.Sequential(
                nn.Linear(shape[0], h1),
                nn.LayerNorm(h1),
                nn.GELU(),
                nn.Dropout(0.22),
                nn.Linear(h1, h2),
                nn.LayerNorm(h2),
                nn.GELU(),
                nn.Dropout(0.20),
                *(
                    [nn.Linear(h2, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.15)]
                    if heavy
                    else []
                ),
                nn.Linear(h2, outputs),
            )
            return
        conv = (nn.Conv1d, nn.Conv2d)[dim - 1]
        norm = (nn.BatchNorm1d, nn.BatchNorm2d)[dim - 1]
        avg = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)[dim - 1]
        stem_stride = 4 if dim == 2 else 2
        stem_kernel = 4 if dim == 2 else 5
        w1, w2, w3 = (40, 112, 352) if heavy else (16, 32, 48)
        self.stem = nn.Sequential(
            conv(
                shape[0],
                w1,
                stem_kernel,
                stride=stem_stride,
                padding=0 if dim == 2 else 2,
            ),
            norm(w1),
            nn.GELU(),
        )
        self.stage = nn.Sequential(
            conv(w1, w2, 3, stride=2, padding=1), norm(w2), nn.GELU()
        )
        self.context = nn.Sequential(
            conv(w2, w3, 3, stride=2, padding=1),
            norm(w3),
            nn.GELU(),
            *[PointResidual(w3, dim) for _ in range(2 if heavy else 1)],
        )
        if not dense:
            self.head = nn.Sequential(
                avg(1),
                nn.Flatten(),
                nn.Linear(w3, w3),
                nn.GELU(),
                nn.Dropout(0.18),
                nn.Linear(w3, outputs),
            )
        else:
            self.skip_decode = nn.Sequential(nn.Conv2d(w3, w2, 1), nn.GELU())
            self.upsample = nn.Sequential(
                nn.Conv2d(w2 * 2, w1, 3, padding=1), nn.GELU()
            )
            self.head = nn.Sequential(
                nn.Conv2d(w1 * 2, w1, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(w1, outputs * 16, 1),
                nn.PixelShuffle(4),
            )

    def forward(self, x):
        if self.dim == 0:
            return self.mlp(x)
        a = self.stem(x)
        b = self.stage(a)
        z = self.context(b)
        if not self.dense:
            return self.head(z)
        z = F.interpolate(
            self.skip_decode(z), size=b.shape[2:], mode="bilinear", align_corners=False
        )
        z = self.upsample(torch.cat((z, b), 1))
        z = F.interpolate(z, size=a.shape[2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat((z, a), 1))


def scores(y, p, kind):
    if kind == "flow":
        delta = (y - p) * 8
        return {"endpoint_error_px": float(np.sqrt((delta**2).sum(axis=1)).mean())}
    if kind == "image":
        pred = np.clip(p, 0, 1)
        per = ((y - pred) ** 2).mean((1, 2, 3))
        return {
            "psnr": float(np.mean(-10 * np.log10(np.maximum(per, 1e-9)))),
            "mae": float(np.abs(y - pred).mean()),
        }
    if kind == "mask":
        pred = p >= 0
        dice = (2 * (pred * y).sum((1, 2, 3)) + 1) / (
            pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1
        )
        points = torch.from_numpy(pred.astype(np.float32))
        target = torch.from_numpy(y.astype(np.float32))

        def edges(a):
            return (a - F.max_pool2d(a, 3, 1, 1).neg()).numpy()

        # Inner boundary is foreground minus a one-pixel erosion.
        pe = (points + F.max_pool2d(-points, 3, 1, 1)).numpy() > 0
        ye = (target + F.max_pool2d(-target, 3, 1, 1)).numpy() > 0
        edge = (2 * (pe & ye).sum((1, 2, 3)) + 1) / (
            pe.sum((1, 2, 3)) + ye.sum((1, 2, 3)) + 1
        )
        return {"dice": float(dice.mean()), "boundary_dice": float(edge.mean())}
    probs = torch.from_numpy(p).softmax(1).numpy()
    result = {"accuracy": float(accuracy_score(y, probs.argmax(1)))}
    if p.shape[1] == 2:
        result.update(
            auroc=float(roc_auc_score(y, probs[:, 1])),
            average_precision=float(average_precision_score(y, probs[:, 1])),
            brier=float(brier_score_loss(y, probs[:, 1])),
        )
    return result


def run(data, rec, seed, folder):
    started = time.perf_counter()
    tr, va, te = split(data, seed)
    kind = data["kind"]
    dense = kind != "classification"
    rng = np.random.default_rng(seed)
    if len(tr) > 3600:
        tr = np.sort(rng.choice(tr, 3600, replace=False))
    if len(va) > 1400:
        va = np.sort(rng.choice(va, 1400, replace=False))
    x = np.asarray(data["x"], np.float32)
    y = np.asarray(data["y"])
    axes = (0, *range(2, x.ndim))
    mean = x[tr].mean(axes, keepdims=True)
    scale = x[tr].std(axes, keepdims=True) + 1e-5

    def tensor(v):
        return torch.from_numpy(
            np.ascontiguousarray(np.clip((v - mean) / scale, -8, 8), dtype=np.float32)
        )

    x0 = tensor(x)
    if x.ndim == 4:
        x0 = x0.contiguous(memory_format=torch.channels_last)
    stress = tensor(data["stress"]) if "stress" in data else None
    targets = torch.as_tensor(y, dtype=torch.float32 if dense else torch.long)
    out = {}
    for heavy in (False, True):
        arm = "heavy" if heavy else "compact"
        torch.manual_seed(seed)
        model = DeepNetwork(x.shape[1:], data["outputs"], dense, heavy)
        if x.ndim == 4:
            model.to(memory_format=torch.channels_last)
        train_ids = tr
        if rec["mode"] == "scale" and not heavy:
            train_ids = np.sort(
                np.random.default_rng(seed + 130).choice(
                    tr, min(len(tr), 650), replace=False
                )
            )
        if rec["mode"] == "fair":
            age_group = (np.asarray(data["age"]) >= 7).astype(int)
            combo = age_group * 2 + y
            counts = np.bincount(combo[train_ids], minlength=4).clip(min=1)
            sample_weight = torch.as_tensor(1 / counts[combo], dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0004 if kind == "flow" else 0.0015,
            weight_decay=0.002,
        )
        max_epochs = 12 if heavy else 8
        best = float("inf")
        epoch_best = 0
        size = 32 if dense else 64

        def loss(pred, label):
            if kind == "image":
                return F.l1_loss(pred, label) + 0.25 * F.mse_loss(pred, label)
            if kind == "mask":
                q = pred.sigmoid()
                dice = (2 * (q * label).sum() + 1) / (q.sum() + label.sum() + 1)
                return F.binary_cross_entropy_with_logits(pred, label) + 1 - dice
            return F.cross_entropy(pred, label)

        def predict(features, ids):
            model.eval()
            with torch.inference_mode():
                return torch.cat(
                    [
                        (
                            model(features[i]).tanh()
                            if kind == "flow"
                            else model(features[i])
                        )
                        for i in np.array_split(ids, max(1, len(ids) // 64 + 1))
                    ]
                )

        trace = []
        for epoch in range(max_epochs):
            model.train()
            for j in torch.randperm(len(train_ids)).split(size):
                ids = train_ids[j.numpy()]
                batch = x0[ids]
                if heavy and rec["mode"] in ("icu", "pair", "diabetes"):
                    batch = batch.clone()
                    batch[::3] = stress[ids[::3]]
                pred = model(batch)
                if kind == "flow":
                    pred = pred.tanh()
                if rec["mode"] == "fair" and heavy:
                    w = sample_weight[ids]
                    value = (
                        F.cross_entropy(pred, targets[ids], reduction="none") * w
                    ).sum() / w.sum()
                else:
                    value = loss(pred, targets[ids])
                optimizer.zero_grad()
                value.backward()
                if kind == "flow":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            v = loss(predict(x0, va), targets[va]).item()
            trace.append(v)
            if v < best - 1e-5:
                epoch_best = epoch + 1
                best = v
                state = {k: w.detach().clone() for k, w in model.state_dict().items()}
            elif epoch + 1 - epoch_best >= 4:
                break
        model.load_state_dict(state)
        pred = predict(x0, te).numpy()
        if rec["mode"] == "mri_freq":
            # Replace measured Fourier lines with those derived from the zero-filled input.
            kspace = np.fft.fftshift(np.fft.fft2(pred, axes=(-2, -1)), axes=(-2, -1))
            known = np.fft.fftshift(
                np.fft.fft2(x[te, :1], axes=(-2, -1)), axes=(-2, -1)
            )
            keep = data["kspace"]
            kspace[:, :, keep, :] = known[:, :, keep, :]
            consistent = np.fft.ifft2(
                np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)
            ).real
            out[arm + "_consistent"] = scores(y[te], consistent, "image")
        result = scores(y[te], pred, kind)
        result["training_records"] = len(train_ids)
        result.update(
            parameters=sum(w.numel() for w in model.parameters()),
            best_epoch=epoch_best,
            epochs_ran=len(trace),
        )
        out[arm] = result
        torch.save(
            {
                "weights": state,
                "validation_loss": trace,
                "training_seed": seed,
                "best_epoch": epoch_best,
            },
            folder / f"{arm}-{seed}.pt",
        )
        if stress is not None:
            out[arm + "_stress"] = scores(y[te], predict(stress, te).numpy(), kind)
        if rec["mode"] == "fair":
            if len(np.unique(y[te])) == 2:
                quart = np.asarray(data["age"])[te] >= 7
                for label, m in [("under_70", ~quart), ("age_70_plus", quart)]:
                    if m.sum() and len(set(y[te][m])) == 2:
                        out[arm + "_" + label] = scores(y[te][m], pred[m], kind)
        print(rec["prof"], rec["real"], seed, arm, result, flush=True)
    if "baseline" in data:
        out["fixed_reference"] = scores(y[te], data["baseline"][te], kind)
    np.savez(
        folder / f"split-{seed}.npz",
        train=tr,
        validation=va,
        test=te,
        mean=mean,
        scale=scale,
    )
    return {
        "sizes": [len(tr), len(va), len(te)],
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    data_root = Path(os.environ.get("PORTFOLIO_DATA", ROOT / "datasets"))
    rec = next(
        r
        for r in json.loads((ROOT / "training-logs/training-batch-79.json").read_text())
        if r["i"] == args.i and r["real"] == args.real
    )
    out = ROOT / rec["folder"]
    folder = out / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    data = load_task(data_root, rec["mode"], args.real)
    runs = {}
    for seed in SEEDS:
        runs[seed] = run(data, rec, seed, folder)
        (folder / "progress.json").write_text(json.dumps(runs, indent=2))
    arms = set.intersection(*(set(r["metrics"]) for r in runs.values()))
    metrics = {
        arm: {
            k: float(np.mean([r["metrics"][arm][k] for r in runs.values()]))
            for k in runs[SEEDS[0]]["metrics"][arm]
        }
        for arm in sorted(arms)
    }
    report = {
        "study": rec,
        "input_shape": list(data["x"].shape[1:]),
        "observations": len(data["y"]),
        "device": "cpu",
        "runs": runs,
        "means": metrics,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
