"""Train a compact residual encoder and compare open-set rejection rules."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import weibull_min
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]


def cifar():
    from io import BytesIO

    from PIL import Image
    from pyarrow.parquet import ParquetFile

    def read(name, rows):
        parquet = ParquetFile(ROOT / f"datasets/cifar100-{name}.parquet")
        assert parquet.metadata.num_rows == rows
        pixels = np.empty((rows, 3, 32, 32), dtype=np.uint8)
        fine_labels = np.empty(rows, dtype=int)
        coarse_labels = np.empty(rows, dtype=int)
        offset = 0
        for batch in parquet.iter_batches(batch_size=512):
            for record in batch.to_pylist():
                with Image.open(BytesIO(record["img"]["bytes"])) as image:
                    pixels[offset] = np.asarray(image.convert("RGB")).transpose(2, 0, 1)
                fine_labels[offset] = record["fine_label"]
                coarse_labels[offset] = record["coarse_label"]
                offset += 1
        assert offset == rows
        return pixels, fine_labels, coarse_labels

    return read("train", 50000), read("test", 10000)


def simulation(seed):
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1, 1, 32, dtype=np.float32)
    u, v = np.meshgrid(grid, grid)

    def part(per_class):
        frames, fine, coarse = [], [], []
        for family in range(4):
            for variant in range(5):
                angle = rng.uniform(-0.30, 0.30, (per_class, 1, 1))
                x = (u - rng.normal(0, 0.06, (per_class, 1, 1))) * np.cos(
                    angle
                ) - v * np.sin(angle)
                y = u * np.sin(angle) + (
                    v - rng.normal(0, 0.06, (per_class, 1, 1))
                ) * np.cos(angle)
                radius = np.hypot(x, y)
                phi = np.arctan2(y, x)
                function = (x, y, radius, phi / np.pi)[family]
                tone = np.sin(
                    function * np.pi * (1.5 + variant * 0.70)
                    + rng.normal(0, 0.4, (per_class, 1, 1))
                )
                envelope = np.clip(1.3 - radius, 0, 1)
                color = rng.uniform(0.6, 1.4, (per_class, 3, 1, 1))
                noise = rng.normal(0, 0.13, (per_class, 3, 32, 32))
                image = np.clip(
                    0.48 + (color * (tone * envelope)[:, None]) * 0.25 + noise, 0, 1
                )
                frames.append((image * 255).astype(np.uint8))
                fine.extend([family * 5 + variant] * per_class)
                coarse.extend([family] * per_class)
        return np.concatenate(frames), np.asarray(fine), np.asarray(coarse)

    return part(650), part(260)


def partitions(y, coarse, future_labels, seed):
    rng = np.random.default_rng(seed)
    known, validation_unknown, test_unknown = [], [], []
    for family in np.unique(coarse):
        names = rng.permutation(np.unique(y[coarse == family]))
        assert len(names) == 5
        known.extend(names[:3].tolist())
        validation_unknown.append(int(names[3]))
        test_unknown.append(int(names[4]))
    known = np.sort(known)
    indices = np.flatnonzero(np.isin(y, known))
    train, val = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=y[indices]
    )
    return {
        "train": train,
        "val_known": val,
        "val_unknown": np.flatnonzero(np.isin(y, validation_unknown)),
        "test_known": np.flatnonzero(np.isin(future_labels, known)),
        "test_unknown": np.flatnonzero(np.isin(future_labels, test_unknown)),
        "known_classes": known,
        "validation_unknown_classes": np.asarray(validation_unknown),
        "test_unknown_classes": np.asarray(test_unknown),
    }


def residual(width):
    return nn.Sequential(
        nn.Conv2d(width, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.GELU(),
        nn.Conv2d(width, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width),
    )


def block(incoming, width):
    return nn.Sequential(
        nn.Conv2d(incoming, width, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.GELU(),
    )


class RecognitionModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.stem = block(3, 48)
        self.block1 = residual(48)
        self.stage2 = block(48, 88)
        self.block2 = residual(88)
        self.stage3 = block(88, 128)
        self.block3 = residual(128)
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        self.head = nn.Linear(128, classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.stage2(F.gelu(h + self.block1(h)))
        h = self.stage3(F.gelu(h + self.block2(h)))
        z = self.project(F.gelu(h + self.block3(h)))
        return self.head(z), z


def decode(raw, indexes, device):
    return (
        torch.from_numpy(raw[indexes])
        .to(device=device, dtype=torch.float32)
        .div_(127.5)
        .sub_(1)
    )


def predict(model, raw, indexes, device, batch_size):
    model.eval()
    predictions, embedding = [], []
    with torch.inference_mode():
        for start in range(0, len(indexes), batch_size):
            logits, z = model(decode(raw, indexes[start : start + batch_size], device))
            predictions.append(logits.softmax(1).cpu().numpy())
            embedding.append(F.normalize(z, dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embedding)


def train(x, y, split, args, output):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train_ids, val_ids = split["train"], split["val_known"]
    model = RecognitionModel(len(split["known_classes"])).to(
        device, memory_format=torch.channels_last
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.006)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    rng = np.random.default_rng(args.seed)
    checkpoint, last = output / "backbone.pt", output / "last-backbone.pt"
    best, logs = -1, []
    if last.exists():
        state = torch.load(last, map_location=device)
        identity = {
            key: val
            for key, val in vars(args).items()
            if key not in ("device", "threads")
        }
        if state["configuration"] != identity:
            raise ValueError(
                "Existing checkpoint uses a different training configuration"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, logs = state["accuracy"], state["logs"]
        rng.bit_generator.state = state["rng"]
    for epoch in range(len(logs) + 1, args.epochs + 1):
        start = time.monotonic()
        torch.manual_seed(args.seed + epoch)
        permutation = rng.permutation(train_ids)
        model.train()
        for offset in range(0, len(permutation), args.batch_size):
            selected = permutation[offset : offset + args.batch_size]
            pixels = decode(x, selected, device)
            if rng.random() < 0.5:
                pixels = pixels.flip(-1)
            pixels = (pixels + torch.randn_like(pixels) * 0.04).clamp(-1, 1)
            logits, _ = model(pixels)
            loss = criterion(logits, torch.as_tensor(y[selected], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        p, _ = predict(model, x, val_ids, device, args.batch_size)
        accuracy = float((p.argmax(1) == y[val_ids]).mean())
        entry = {
            "epoch": epoch,
            "known_validation_accuracy": accuracy,
            "seconds": time.monotonic() - start,
        }
        logs.append(entry)
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), checkpoint.with_suffix(".tmp"))
            checkpoint.with_suffix(".tmp").replace(checkpoint)
        identity = {
            key: val
            for key, val in vars(args).items()
            if key not in ("device", "threads")
        }
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "configuration": identity,
                "accuracy": best,
                "logs": logs,
                "rng": rng.bit_generator.state,
            },
            last.with_suffix(".tmp"),
        )
        last.with_suffix(".tmp").replace(last)
        print(entry, flush=True)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model, logs


def fit_tails(features, labels, seed):
    rng = np.random.default_rng(seed)
    anchors = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == c), 24, replace=False)
            for c in np.unique(labels)
        ]
    )
    margin_refs = []
    for i in range(0, len(anchors), 32):
        selected = anchors[i : i + 32]
        distance = np.sqrt(np.maximum(0, 2 - 2 * features[selected] @ features.T))
        distance[labels[selected, None] == labels[None, :]] = np.inf
        margin_refs.extend(np.partition(distance, 47, axis=1)[:, :48] / 2)
    parameters = np.asarray(
        [weibull_min.fit(np.maximum(row, 1e-5), floc=0) for row in margin_refs]
    )
    return features[anchors], labels[anchors], parameters[:, 0], parameters[:, 2]


def class_evidence(z, anchors, labels, shape, scale, n_classes):
    outputs = []
    for start in range(0, len(z), 256):
        distance = np.sqrt(np.maximum(0, 2 - 2 * z[start : start + 256] @ anchors.T))
        survival = np.exp(-np.power(distance / scale, shape))
        outputs.append(
            np.column_stack(
                [survival[:, labels == c].max(axis=1) for c in range(n_classes)]
            )
        )
    return np.concatenate(outputs)


def performance(known_scores, unknown_scores, correct, threshold):
    labels = np.r_[
        np.ones(len(known_scores), dtype=int), np.zeros(len(unknown_scores), dtype=int)
    ]
    scores = np.r_[known_scores, unknown_scores]
    order = np.argsort(-scores, kind="stable")
    ends = np.flatnonzero(np.r_[scores[order][:-1] != scores[order][1:], True])
    correct_known = np.r_[correct, np.zeros(len(unknown_scores), dtype=bool)]
    correct_rate = np.r_[0, correct_known[order].cumsum()[ends] / len(correct)]
    false_accept = np.r_[0, (labels[order] == 0).cumsum()[ends] / len(unknown_scores)]
    return {
        "unknown_auroc": float(roc_auc_score(1 - labels, -scores)),
        "unknown_ap": float(average_precision_score(1 - labels, -scores)),
        "known_accuracy": float(correct.mean()),
        "oscr": float(np.trapezoid(correct_rate, false_accept)),
        "threshold": float(threshold),
        "known_acceptance": float((known_scores >= threshold).mean()),
        "unknown_false_acceptance": float((unknown_scores >= threshold).mean()),
    }


def main(args):
    torch.set_num_threads(args.threads)
    source, official_test = cifar() if args.real else simulation(args.seed)
    x, y, families = source
    tx, ty, _ = official_test
    splits = partitions(y, families, ty, args.seed)
    indices = {label: i for i, label in enumerate(splits["known_classes"])}
    masked = np.asarray([indices.get(label, -1) for label in y])
    truth = np.asarray([indices.get(label, -1) for label in ty])
    test_known = splits["test_known"]
    output = ROOT / (
        "real-data-projects/scheirer-cifar-open-set"
        if args.real
        else "professor-aligned/scheirer-extreme-tail-recognition"
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "splits.npz", **splits)
    model, history = train(x, masked, splits, args, output)
    device = torch.device(args.device)
    _, reference = predict(model, x, splits["train"], device, args.batch_size)
    tails = fit_tails(reference, masked[splits["train"]], args.seed)
    np.savez_compressed(
        output / "weibull-tails.npz",
        anchors=tails[0],
        classes=tails[1],
        shapes=tails[2],
        scales=tails[3],
    )
    evaluation = {}
    for key in ("val_known", "val_unknown", "test_known", "test_unknown"):
        data = tx if key.startswith("test") else x
        soft, z = predict(model, data, splits[key], device, args.batch_size)
        evidence = class_evidence(z, *tails, len(indices))
        evaluation[key] = {
            "softmax": (soft.max(1), soft.argmax(1)),
            "tail_inclusion": (evidence.max(1), evidence.argmax(1)),
        }
    results = {
        "configuration": vars(args),
        "source_train_images": len(x),
        "source_test_images": len(tx),
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "counts": {key: len(value) for key, value in splits.items()},
        "methods": {},
    }
    for mode in ("softmax", "tail_inclusion"):
        scores, predicted = evaluation["test_known"][mode]
        unknown, _ = evaluation["test_unknown"][mode]
        valid_known, valid_pred = evaluation["val_known"][mode]
        valid_unknown, _ = evaluation["val_unknown"][mode]
        threshold = np.quantile(valid_known, 0.05)
        results.setdefault("validation", {})[mode] = performance(
            valid_known,
            valid_unknown,
            valid_pred == masked[splits["val_known"]],
            threshold,
        )
        results["methods"][mode] = performance(
            scores, unknown, predicted == truth[test_known], threshold
        )
        np.savez_compressed(
            output / f"{mode}-test.npz",
            known_scores=scores,
            unknown_scores=unknown,
            known_predicted=predicted,
            known_labels=truth[test_known],
        )
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(results["methods"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=34)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
