import argparse
import csv
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
SIZE = 96
ALPHAS = (0.25, 0.375, 0.5, 0.625, 0.75)


def busbra():
    images, masks, groups, ids = [], [], [], []
    with ZipFile(ROOT / "datasets/busbra.zip") as archive:
        rows = csv.DictReader(io.TextIOWrapper(archive.open("BUSBRA/bus_data.csv")))
        names = set(archive.namelist())
        for row in rows:
            sample = row["ID"]
            image_name = f"BUSBRA/Images/{sample}.png"
            options = [
                f"BUSBRA/Masks/{name}.png"
                for name in (sample, sample.replace("bus_", "mask_"), f"{sample}_mask")
            ]
            mask_name = next((name for name in options if name in names), None)
            if mask_name is None:
                raise ValueError(f"Missing mask for {sample}")
            with (
                Image.open(archive.open(image_name)) as image,
                Image.open(archive.open(mask_name)) as mask,
            ):
                images.append(
                    np.asarray(
                        image.convert("L").resize(
                            (SIZE, SIZE), Image.Resampling.BILINEAR
                        ),
                        dtype=np.float32,
                    )
                    / 255
                )
                masks.append(
                    np.asarray(
                        mask.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)
                    )
                    > 0
                )
            groups.append(int(row["Case"]))
            ids.append(sample)
    return images, masks, groups, ids


def phantoms():
    rng = np.random.default_rng(2027)
    y_grid, x_grid = np.mgrid[:SIZE, :SIZE]
    images, masks = [], []
    for _ in range(1800):
        x = x_grid - rng.uniform(22, 74)
        y = y_grid - rng.uniform(22, 74)
        angle = rng.uniform(0, np.pi)
        u = (x * np.cos(angle) + y * np.sin(angle)) / rng.uniform(7, 20)
        v = (-x * np.sin(angle) + y * np.cos(angle)) / rng.uniform(7, 20)
        radius = np.hypot(u, v) + 0.10 * np.cos(5 * np.arctan2(v, u))
        signal = gaussian_filter((radius <= 1).astype(float), 2.0)
        texture = gaussian_filter(rng.normal(size=signal.shape), 3.0)
        images.append(
            np.clip(
                (0.55 - rng.uniform(0.13, 0.42) * signal + texture * 0.7)
                * rng.lognormal(0, 0.23, signal.shape),
                0,
                1,
            )
        )
        masks.append(radius <= rng.uniform(0.84, 1.16))
    ids = list(range(len(images)))
    return images, masks, ids, ids


def split_samples(groups):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2027)
    train_val, test = next(outer.split(groups, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2028)
    train, val = next(inner.split(train_val, groups=groups[train_val]))
    splits = {"train": train_val[train], "val": train_val[val], "test": test}
    assert sum(len(set(groups[index])) for index in splits.values()) == len(set(groups))
    return splits


def conv(incoming, width, stride=1):
    return nn.Sequential(
        nn.Conv2d(incoming, width, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, width),
        nn.SiLU(),
    )


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.first = conv(width, width)
        self.last = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(8, width)
        )

    def forward(self, x):
        return F.silu(x + self.last(self.first(x)))


class Decoder(nn.Module):
    def __init__(self, incoming, width):
        super().__init__()
        self.block = nn.Sequential(conv(incoming + width, width), Residual(width))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class Segmenter(nn.Module):
    def __init__(self, conditional):
        super().__init__()
        self.enc1 = nn.Sequential(conv(1, 24, 2), Residual(24))
        self.enc2 = nn.Sequential(conv(24, 48, 2), Residual(48))
        self.enc3 = nn.Sequential(conv(48, 80, 2), Residual(80))
        self.enc4 = nn.Sequential(conv(80, 128, 2), Residual(128), Residual(128))
        self.condition = (
            nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 256))
            if conditional
            else None
        )
        self.dec3 = Decoder(128, 80)
        self.dec2 = Decoder(80, 48)
        self.dec1 = Decoder(48, 24)
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, image, alpha):
        first = self.enc1(image)
        second = self.enc2(first)
        third = self.enc3(second)
        latent = self.enc4(third)
        if self.condition is not None:
            scale, bias = self.condition((alpha[:, None] - 0.5) * 4).chunk(2, dim=1)
            latent = latent * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        logits = self.head(
            self.dec1(self.dec2(self.dec3(latent, third), second), first)
        )
        return F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False
        )


def objective(logits, mask, alpha):
    p = logits.sigmoid()
    tp = (p * mask).sum((1, 2, 3))
    fp = (p * (1 - mask)).sum((1, 2, 3))
    fn = ((1 - p) * mask).sum((1, 2, 3))
    tversky = (tp + 1) / (tp + alpha * fp + (1 - alpha) * fn + 1)
    return (1 - tversky).mean() + 0.35 * F.binary_cross_entropy_with_logits(
        logits, mask
    )


def augment(image, mask):
    dims = tuple(axis for axis in (-1, -2) if torch.rand(()).item() < 0.5)
    if dims:
        image, mask = image.flip(dims), mask.flip(dims)
    k = int(torch.randint(0, 4, ()).item())
    image, mask = image.rot90(k, (-2, -1)), mask.rot90(k, (-2, -1))
    gain = torch.rand((len(image), 1, 1, 1), device=image.device) * 0.3 + 0.85
    return (image * gain + torch.randn_like(image) * 0.025).clamp(0, 1), mask


def evaluate(model, loader, device, alphas):
    model.eval()
    probability, targets = [], []
    with torch.inference_mode():
        for image, mask in loader:
            image = image.to(device)
            conditional = [
                model(image, torch.full((len(image),), a, device=device))
                .sigmoid()
                .cpu()
                for a in alphas
            ]
            probability.append(torch.stack(conditional, dim=1).numpy())
            targets.append(mask.numpy())
    return np.concatenate(probability, axis=0), np.concatenate(targets, axis=0)


def dice(probability, mask):
    axes = (1, 2, 3)
    predicted = probability >= 0.5
    return (2 * (predicted * mask).sum(axes) + 1e-6) / (
        predicted.sum(axes) + mask.sum(axes) + 1e-6
    )


def report(probabilities, mask):
    averaged = probabilities.mean(axis=1)
    brier = (averaged - mask) ** 2
    boundary = np.stack(
        [
            binary_dilation(m[0], iterations=3) ^ binary_erosion(m[0], iterations=3)
            for m in mask
        ]
    )[:, None]
    return {
        "mean_dice": float(dice(averaged, mask).mean()),
        "brier": float(brier.mean()),
        "boundary_brier": float(brier[boundary].mean()),
        "foreground_fraction_by_alpha": (probabilities >= 0.5)
        .mean((0, 2, 3, 4))
        .tolist(),
    }


def paired_ci(delta, groups):
    rng = np.random.default_rng(41)
    unique, index = np.unique(groups, return_inverse=True)
    totals = np.bincount(index, weights=delta)
    counts = np.bincount(index)
    draws = rng.integers(len(unique), size=(3000, len(unique)))
    return np.quantile(
        totals[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]
    ).tolist()


def write_json(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = ROOT / (
        "real-data-projects/golland-busbra-contour-probability"
        if args.real
        else "professor-aligned/golland-loss-conditioned-segmentation"
    )
    output.mkdir(parents=True, exist_ok=True)
    images, masks, groups, ids = busbra() if args.real else phantoms()
    groups = np.asarray(groups)
    splits = split_samples(groups)
    images = torch.tensor(np.asarray(images), dtype=torch.float32).unsqueeze(1)
    masks = torch.tensor(np.asarray(masks), dtype=torch.float32).unsqueeze(1)
    assert images.shape == masks.shape and bool(masks.any(dim=(1, 2, 3)).all())
    write_json(
        output / "splits.json",
        {
            name: [{"id": ids[i], "group": int(groups[i])} for i in indexes]
            for name, indexes in splits.items()
        },
    )
    metrics, predictions = {}, {}
    for conditional in (False, True):
        name = "loss_conditioned" if conditional else "fixed_tversky"
        torch.manual_seed(args.seed)
        loaders = {
            split: DataLoader(
                TensorDataset(images[indexes], masks[indexes]),
                batch_size=args.batch_size,
                shuffle=split == "train",
                generator=torch.Generator().manual_seed(args.seed),
            )
            for split, indexes in splits.items()
        }
        model = Segmenter(conditional).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0.00005
        )
        checkpoint = output / f"{name}-{args.seed}.pt"
        last = output / f"{name}-{args.seed}-last.pt"
        progress = output / f"{name}-{args.seed}-progress.json"
        alphas = ALPHAS if conditional else (0.5,)
        history = []
        best = -1
        if last.exists():
            saved = torch.load(last, map_location=device)
            if saved["configuration"] != vars(args):
                raise ValueError("Resume arguments differ from the saved configuration")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            lr.load_state_dict(saved["lr"])
            history = saved["history"]
            best = saved["best"]
        for epoch in range(len(history) + 1, args.epochs + 1):
            started = time.monotonic()
            torch.manual_seed(args.seed + epoch)
            loaders["train"].generator.manual_seed(args.seed + epoch)
            model.train()
            total = 0.0
            for image, mask in loaders["train"]:
                image, mask = augment(image.to(device), mask.to(device))
                alpha = (
                    torch.rand(len(image), device=device) * 0.5 + 0.25
                    if conditional
                    else torch.full((len(image),), 0.5, device=device)
                )
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(image, alpha), mask, alpha)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(image)
            lr.step()
            prob, target = evaluate(model, loaders["val"], device, alphas)
            score = float(dice(prob.mean(1), target).mean())
            row = {
                "epoch": epoch,
                "train_loss": total / len(splits["train"]),
                "val_dice": score,
                "seconds": round(time.monotonic() - started, 1),
            }
            history.append(row)
            if score > best:
                best = score
                torch.save(model.state_dict(), checkpoint.with_suffix(".tmp"))
                checkpoint.with_suffix(".tmp").replace(checkpoint)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr": lr.state_dict(),
                    "history": history,
                    "best": best,
                    "configuration": vars(args),
                },
                last.with_suffix(".tmp"),
            )
            last.with_suffix(".tmp").replace(last)
            write_json(progress, history)
            print(name, row, flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        prob, target = evaluate(model, loaders["test"], device, alphas)
        predictions[name] = prob.mean(1)
        metrics[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "alphas": list(alphas),
            "best_validation_dice": best,
            **report(prob, target),
        }
    difference = dice(predictions["loss_conditioned"], target) - dice(
        predictions["fixed_tversky"], target
    )
    write_json(
        output / f"results-{args.seed}.json",
        {
            "configuration": vars(args),
            "cohort": {
                name: {"images": len(indexes), "groups": len(set(groups[indexes]))}
                for name, indexes in splits.items()
            },
            "models": metrics,
            "dice_difference_paired_group_bootstrap_95_ci": paired_ci(
                difference, groups[splits["test"]]
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
