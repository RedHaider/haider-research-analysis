import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", ROOT / "data"))
torch.set_num_threads(2)
SEEDS = (13, 37, 61)


class Network(nn.Module):
    def __init__(self, channels, classes, dim, large):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv1d
        norm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm1d
        pool = nn.MaxPool2d if dim == 2 else nn.MaxPool1d
        avg = nn.AdaptiveAvgPool2d if dim == 2 else nn.AdaptiveAvgPool1d
        widths = (24, 48, 72, 96) if large else (12, 24)
        blocks = []
        for width in widths:
            depth = 2 if large else 1
            for _ in range(depth):
                blocks.extend(
                    [
                        conv(channels, width, 3, padding=1, bias=False),
                        norm(width),
                        nn.GELU(),
                    ]
                )
                channels = width
            blocks.append(pool(2, ceil_mode=True))
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            avg(1), nn.Flatten(), nn.Dropout(0.15), nn.Linear(channels, classes)
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class GlucoseNet(nn.Module):
    def __init__(self, channels, large):
        super().__init__()
        self.large = large
        if large:
            self.seq = nn.GRU(channels, 96, 2, dropout=0.1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
        else:
            self.head = nn.Sequential(
                nn.Linear(channels, 24), nn.GELU(), nn.Linear(24, 1)
            )

    def forward(self, x):
        if self.large:
            z, _ = self.seq(x.transpose(1, 2))
            return self.head(z[:, -1])
        return self.head(x[:, :, -1])


def split(y, g, seed, given):
    if given is not None:
        test = np.flatnonzero(given)
        pool = np.flatnonzero(~given)
    else:
        pool, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.23, random_state=seed).split(
                y, y, g
            )
        )
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=seed + 21).split(
            y[pool], y[pool], g[pool]
        )
    )
    tr, va = pool[a], pool[b]
    assert not (
        set(g[tr]) & set(g[va])
        or set(g[va]) & set(g[test])
        or set(g[tr]) & set(g[test])
    )
    return tr, va, test


def measure(y, p, classes):
    if classes == 1:
        return {"mae": float(mean_absolute_error(y, p[:, 0] * 100))}
    if classes == 2:
        return {
            "auroc": float(roc_auc_score(y, p[:, 1])),
            "average_precision": float(average_precision_score(y, p[:, 1])),
            "accuracy": float(np.mean(p.argmax(1) == y)),
        }
    return {"accuracy": float(np.mean(p.argmax(1) == y))}


def train(x, y, tr, va, te, classes, seed, large, epochs):
    torch.manual_seed(seed)
    model = (
        GlucoseNet(x.shape[1], large)
        if classes == 1
        else Network(x.shape[1], classes, x.ndim - 2, large)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    objective = nn.HuberLoss(delta=0.35) if classes == 1 else nn.CrossEntropyLoss()
    best = float("inf")
    batch_size = 48
    target = y.float() / 100 if classes == 1 else y.long()
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(tr)).split(batch_size):
            ids = tr[idx.numpy()]
            pred = model(x[ids])
            loss = objective(pred.flatten() if classes == 1 else pred, target[ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            pred = torch.cat(
                [model(x[ids]) for ids in np.array_split(va, max(1, len(va) // 96 + 1))]
            )
            score = objective(
                pred.flatten() if classes == 1 else pred, target[va]
            ).item()
        if score < best:
            best = score
            best_epoch = epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    folder = ROOT / "checkpoints"
    folder.mkdir(exist_ok=True)
    torch.save(state, folder / f"{'medium' if large else 'compact'}-{seed}.pt")

    def predict(ids):
        with torch.inference_mode():
            pred = torch.cat(
                [model(x[p]) for p in np.array_split(ids, max(1, len(ids) // 96 + 1))]
            )
            return (pred if classes == 1 else pred.softmax(1)).numpy()

    return (
        predict(va),
        predict(te),
        sum(p.numel() for p in model.parameters()),
        best_epoch,
    )


def run(raw, seed):
    x, y, g, given, classes, cascade = raw
    tr, va, te = split(y, g, seed, given)
    shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
    axes = (0, *range(2, x.ndim))
    mean = x[tr].mean(axis=axes).reshape(shape)
    std = x[tr].std(axis=axes).reshape(shape) + 1e-5
    x = torch.as_tensor(np.clip((x - mean) / std, -8, 8), dtype=torch.float32)
    labels = torch.from_numpy(y)
    out = {}
    predictions = []
    for large in (False, True):
        name = "medium" if large else "compact"
        p, v, size, epoch = train(
            x, labels, tr, va, te, classes, seed, large, 9 if large else 7
        )
        out[name] = {
            **measure(y[te], v, classes),
            "parameters": size,
            "selected_epoch": epoch,
        }
        predictions.append((p, v))
    np.savez(
        ROOT / "checkpoints" / f"split-{seed}.npz",
        train=tr,
        validation=va,
        test=te,
        channel_mean=mean,
        channel_scale=std,
    )
    if cascade:
        p1, t1 = predictions[0]
        p2, t2 = predictions[1]
        target = measure(y[va], p2, classes)["accuracy"] - 0.015
        for threshold in (0.60, 0.70, 0.80, 0.90, 0.95, 1.01):
            combined = np.where((p1.max(1) >= threshold)[:, None], p1, p2)
            if measure(y[va], combined, classes)["accuracy"] >= target:
                break
        defer = t1.max(1) < threshold
        out["cascade"] = {
            **measure(y[te], np.where(defer[:, None], t2, t1), classes),
            "medium_fraction": float(defer.mean()),
            "threshold": threshold,
        }
    return {
        "sizes": [len(tr), len(va), len(te)],
        "groups": [len(set(g[a])) for a in (tr, va, te)],
        "metrics": out,
    }


import io
import zipfile

from scipy.io import wavfile
from scipy.signal import stft


def dataset():
    chosen = ("down", "go", "left", "up")
    x = []
    y = []
    g = []
    with zipfile.ZipFile(DATA / "mini-speech-commands.zip") as z:
        for path in sorted(z.namelist()):
            item = Path(path)
            label = item.parent.name
            if (
                label not in chosen
                or item.suffix != ".wav"
                or item.name.startswith(".")
            ):
                continue
            sr, w = wavfile.read(io.BytesIO(z.read(path)))
            assert sr == 16000
            w = w.astype(np.float32) / 32768
            w = np.pad(w[:16000], (0, max(0, 16000 - len(w))))
            _, _, s = stft(w, fs=16000, nperseg=256, noverlap=128)
            np.unique(np.rint(np.geomspace(1, 129, 34)).astype(int))
            groups = np.array_split(np.log1p(np.abs(s) * 200), 32, axis=0)
            feature = np.stack(
                [
                    np.array([p.mean() for p in np.array_split(b, 32, axis=1)])
                    for b in groups
                ]
            )
            x.append(feature)
            y.append(chosen.index(label))
            g.append(item.stem.split("_nohash_")[0])
    assert len(x) >= 3500
    return (
        np.asarray(x, dtype=np.float32)[:, None],
        np.asarray(y),
        np.asarray(g),
        None,
        4,
        True,
    )


if __name__ == "__main__":
    data = dataset()
    runs = {s: run(data, s) for s in SEEDS}
    means = {
        arm: {
            name: float(np.mean([r["metrics"][arm][name] for r in runs.values()]))
            for name in row
        }
        for arm, row in runs[SEEDS[0]]["metrics"].items()
    }
    (ROOT / "results.json").write_text(
        json.dumps({"runs": runs, "means": means}, indent=2)
    )
    print(json.dumps(means), flush=True)
