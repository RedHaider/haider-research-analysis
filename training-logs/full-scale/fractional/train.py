"""Forecast multichannel sequences with causal fractional-memory features."""

import argparse
import itertools
import json
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import fftconvolve
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
CHANNELS = ("C3", "C4", "CZ", "F3", "F4", "FZ", "P3", "P4", "PZ")
ORDERS = (0.15, 0.35, 0.60, 0.85)
TAPS = np.array([0, 1, 2, 3, 5, 8, 12, 20, 32, 48, 64, 100, 160, 230])
HORIZONS = np.array([1, 4, 8, 16])
MEMORY = 240


def fields(stream, channels, width):
    data = stream.read(channels * width)
    if len(data) != channels * width:
        raise ValueError("Incomplete EDF channel header")
    return [
        data[i : i + width].decode("ascii").strip() for i in range(0, len(data), width)
    ]


def read_edf(path):
    with path.open("rb") as stream:
        header = stream.read(256)
        offset = int(header[184:192])
        n = int(header[236:244])
        duration = float(header[244:252])
        channels = int(header[252:256])
        labels = [name.rstrip(". ").upper() for name in fields(stream, channels, 16)]
        fields(stream, channels, 80)
        units = fields(stream, channels, 8)
        physical_min = np.array(fields(stream, channels, 8), dtype=np.float32)
        physical_max = np.array(fields(stream, channels, 8), dtype=np.float32)
        digital_min = np.array(fields(stream, channels, 8), dtype=np.float32)
        digital_max = np.array(fields(stream, channels, 8), dtype=np.float32)
        fields(stream, channels, 80)
        counts = np.array(fields(stream, channels, 8), dtype=int)
    selected = [labels.index(label) for label in CHANNELS]
    rate = counts[selected[0]] / duration
    if duration != 1 or rate != 160 or not np.all(counts[selected] == rate):
        return (
            None  # The public corpus contains a few recordings at inconsistent rates.
        )
    if any(units[i].lower() != "uv" for i in selected):
        raise ValueError(f"Unexpected electrode unit in {path.name}")
    begins = np.r_[0, counts.cumsum()]
    rows = np.memmap(
        path, mode="r", dtype="<i2", offset=offset, shape=(n, int(counts.sum()))
    )
    data = np.stack(
        [rows[:, begins[i] : begins[i + 1]].reshape(-1) for i in selected], axis=1
    ).astype(np.float32)
    scale = (physical_max - physical_min)[selected] / (digital_max - digital_min)[
        selected
    ]
    return (data - digital_min[selected]) * scale + physical_min[selected]


def eeg_segments():
    folder = ROOT / "datasets/eegmmidb"
    paths = sorted(folder.glob("S???R04.edf"))
    if len(paths) != 109:
        raise ValueError(
            f"Expected all 109 PhysioNet R04 recordings, found {len(paths)}"
        )
    segments, excluded = [], []
    for path in paths:
        values = read_edf(path)
        if values is None:
            excluded.append(path.name)
            continue
        segments.append((path.stem[:4], path.stem, values))
    return segments, {
        "rate_hz": 160,
        "channels": list(CHANNELS),
        "excluded_source_files": excluded,
    }


def wisdm_segments():
    with tarfile.open(ROOT / "datasets/wisdm-activity-v1.1.tar.gz", "r:gz") as archive:
        source = next(member for member in archive if member.name.endswith("_raw.txt"))
        with archive.extractfile(source) as raw:
            cleaned = (
                raw.read().replace(b"\n", b"").replace(b"\r", b"").replace(b";", b"\n")
            )
    from io import BytesIO

    fields = ["subject", "activity", "time", "x", "y", "z"]
    table = pd.read_csv(BytesIO(cleaned), names=fields, on_bad_lines="skip")
    for name in ("subject", "time", "x", "y", "z"):
        table[name] = pd.to_numeric(table[name], errors="coerce")
    timestamp = ["subject", "activity", "time"]
    table = (
        table.dropna()
        .loc[lambda frame: frame.time > 0]
        .drop_duplicates(timestamp, keep=False)
        .sort_values(timestamp, kind="stable")
    )
    time = table.time.to_numpy()
    actor = table.subject.to_numpy()
    activity = table.activity.to_numpy()
    dt = np.diff(time)
    boundaries = np.r_[
        0,
        np.flatnonzero(
            (dt <= 0)
            | (dt > 200_000_000)
            | (actor[:-1] != actor[1:])
            | (activity[:-1] != activity[1:])
        )
        + 1,
        len(table),
    ]
    values = table[["x", "y", "z"]].to_numpy(dtype=np.float32)
    segments = []
    for start, end in itertools.pairwise(boundaries):
        if end - start > MEMORY + HORIZONS.max():
            segments.append(
                (
                    str(int(actor[start])),
                    f"{int(actor[start])}-{start}-{activity[start]}",
                    values[start:end],
                )
            )
    if len({group for group, _, _ in segments}) < 30:
        raise ValueError("WISDM parsing did not preserve the expected subject coverage")
    return segments, {
        "rate_hz_nominal": 20,
        "channels": ["x", "y", "z"],
        "parsed_rows": len(table),
    }


def gl_weights(order):
    result = np.empty(MEMORY, dtype=np.float32)
    result[0] = 1
    for k in range(1, len(result)):
        result[k] = result[k - 1] * (k - 1 - order) / k
    return result[:, None]


def collect(segments, stride):
    channels, fractional, targets, groups, identifiers = [], [], [], [], []
    for person, run, samples in segments:
        at = np.arange(MEMORY, len(samples) - HORIZONS.max(), stride)
        if not len(at):
            continue
        channels.append(samples[at[:, None] - TAPS].reshape(len(at), -1))
        bank = [
            fftconvolve(samples, gl_weights(alpha), axes=0, mode="full")[at]
            for alpha in ORDERS
        ]
        fractional.append(np.column_stack(bank))
        targets.append(samples[at[:, None] + HORIZONS].reshape(len(at), -1))
        groups.extend([person] * len(at))
        identifiers.extend([f"{run}:{t}" for t in at])
    return (
        np.concatenate(channels),
        np.concatenate(fractional),
        np.concatenate(targets),
        np.asarray(groups),
        np.asarray(identifiers),
    )


def groupsplit(groups, seed):
    trainval, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            groups, groups=groups
        )
    )
    fit, valid = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1).split(
            trainval, groups=groups[trainval]
        )
    )
    result = trainval[fit], trainval[valid], test
    assert sum(len(np.unique(groups[indices])) for indices in result) == len(
        np.unique(groups)
    )
    return result


class Forecaster(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(inputs, 256), nn.LayerNorm(256), nn.SiLU())
        self.residual = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 256),
        )
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(256, outputs))

    def forward(self, x):
        h = self.input(x)
        return self.head(h + self.residual(h))


def errors(target, prediction):
    spread = (prediction - target).reshape(len(target), len(HORIZONS), -1)
    return {
        "rmse": float(np.sqrt(np.mean(spread**2))),
        "rmse_by_horizon": np.sqrt(np.mean(spread**2, axis=(0, 2))).tolist(),
        "mae_by_horizon": np.mean(np.abs(spread), axis=(0, 2)).tolist(),
    }


def fit_mlp(features, y, splits, args, checkpoint):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, valid, test = splits
    model = Forecaster(features.shape[1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.004)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    x, target = torch.from_numpy(features), torch.from_numpy(y)
    best, log = (float("inf"), 0), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in np.array_split(rng.permutation(train), max(1, len(train) // 384)):
            loss = F.smooth_l1_loss(model(x[batch]), target[batch], beta=0.35)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        schedule.step()
        model.eval()
        with torch.inference_mode():
            val_rmse = errors(y[valid], model(x[valid]).numpy())["rmse"]
        log.append(val_rmse)
        if val_rmse < best[0]:
            best = (val_rmse, epoch)
            torch.save(model.state_dict(), checkpoint)
        if epoch - best[1] >= 5:
            break
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    with torch.inference_mode():
        out = model(x[test]).numpy()
    return out, {
        "selected_epoch": best[1],
        "validation_rmse": log,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def normalized(values, train):
    mean = values[train].mean(axis=0)
    scale = values[train].std(axis=0).clip(min=1e-5)
    return np.clip((values - mean) / scale, -12, 12), mean, scale


def main(args):
    torch.set_num_threads(1)
    collection, source = wisdm_segments() if args.real else eeg_segments()
    features, memory, targets, groups, ids = collect(collection, args.stride)
    splits = groupsplit(groups, args.seed)
    train, valid, test = splits
    output = ROOT / (
        "real-data-projects/bogdan-wisdm-fractional-forecast"
        if args.real
        else "professor-aligned/bogdan-eeg-fractional-forecast"
    )
    output.mkdir(parents=True, exist_ok=True)
    x, x_mean, x_scale = normalized(features, train)
    z, z_mean, z_scale = normalized(memory, train)
    y_mean, y_scale = targets[train].mean(0), targets[train].std(0).clip(min=1e-5)
    y = (targets - y_mean) / y_scale
    np.savez_compressed(
        output / "preprocessing.npz",
        x_mean=x_mean,
        x_scale=x_scale,
        z_mean=z_mean,
        z_scale=z_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        train=ids[train],
        validation=ids[valid],
        test=ids[test],
        groups=groups,
        horizons=HORIZONS,
        fractional_orders=ORDERS,
        taps=TAPS,
    )
    channels = targets.shape[1] // len(HORIZONS)
    persistence = np.tile(features[test, :channels], (1, len(HORIZONS)))
    metrics = {"persistence": errors(targets[test], persistence)}
    predictions = {"persistence": persistence}
    for name, xvars in (
        ("short_ridge", x[:, : channels * 6]),
        ("long_ridge", x),
        ("fractional_ridge", np.column_stack([x, z])),
    ):
        ridge = Ridge(alpha=5).fit(xvars[train], y[train])
        predicted = ridge.predict(xvars[test]) * y_scale + y_mean
        metrics[name] = errors(targets[test], predicted)
        predictions[name] = predicted
        joblib.dump(ridge, output / f"{name}.joblib", compress=3)
    for name, xvars in (("long_mlp", x), ("fractional_mlp", np.column_stack([x, z]))):
        predicted, trace = fit_mlp(xvars, y, splits, args, output / f"{name}.pt")
        predictions[name] = predicted * y_scale + y_mean
        metrics[name] = {**errors(targets[test], predictions[name]), **trace}
        print(
            name,
            {
                key: value
                for key, value in metrics[name].items()
                if key != "validation_rmse"
            },
            flush=True,
        )
    np.savez_compressed(output / "held-out.npz", observed=targets[test], **predictions)
    outcome = {
        "configuration": vars(args),
        "source": source,
        "source_rows_in_contiguous_runs": sum(len(value) for _, _, value in collection),
        "splits": {
            key: {"windows": len(ix), "subjects": len(np.unique(groups[ix]))}
            for key, ix in zip(("train", "validation", "test"), splits)
        },
        "horizons": HORIZONS.tolist(),
        "methods": metrics,
    }
    (output / "results.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
