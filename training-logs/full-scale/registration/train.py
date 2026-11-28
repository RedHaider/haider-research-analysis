"""Learn dense and integrated deformations for three-dimensional atlas transfer."""

import argparse
import gzip
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
SHAPE = (24, 26, 30)


def load_mri():
    import nibabel as nib

    with tarfile.open(ROOT / "datasets/msd-hippocampus.tar", "r") as archive:
        images = {
            Path(entry.name).name: entry
            for entry in archive.getmembers()
            if "/imagesTr/" in entry.name
            and entry.name.endswith(".nii.gz")
            and not Path(entry.name).name.startswith("._")
        }
        labels = {
            Path(entry.name).name: entry
            for entry in archive.getmembers()
            if "/labelsTr/" in entry.name
            and entry.name.endswith(".nii.gz")
            and not Path(entry.name).name.startswith("._")
        }
        identifiers = sorted(images.keys() & labels.keys())
        if len(identifiers) < 240:
            raise ValueError(
                f"Unexpected number of labeled MRI volumes: {len(identifiers)}"
            )

        def read(member):
            with archive.extractfile(member) as stream:
                vol = nib.Nifti1Image.from_bytes(gzip.decompress(stream.read()))
            return nib.as_closest_canonical(vol).get_fdata(dtype=np.float32)

        volumes, masks = [], []
        for name in identifiers:
            image = read(images[name])
            target = read(labels[name])
            if image.shape != target.shape:
                raise ValueError(f"Image and label shape differs for {name}")
            q1, q99 = np.percentile(image, (1, 99))
            intensity = np.clip((image - q1) / max(q99 - q1, 1e-4), 0, 1)
            volumes.append(
                F.interpolate(
                    torch.tensor(intensity, dtype=torch.float32)[None, None],
                    size=SHAPE,
                    mode="trilinear",
                    align_corners=True,
                )[0]
            )
            masks.append(
                F.interpolate(
                    torch.tensor(target)[None, None], size=SHAPE, mode="nearest"
                )[0]
            )
    return torch.stack(volumes), torch.stack(masks), identifiers


def phantoms(seed):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*(np.linspace(-1, 1, n) for n in SHAPE), indexing="ij")
    volumes, masks = [], []
    for _ in range(900):
        X = (x - rng.uniform(-0.16, 0.16)) + rng.uniform(-0.25, 0.25) * y
        Y = (y - rng.uniform(-0.16, 0.16)) + rng.uniform(-0.2, 0.2) * x
        Z = z - rng.uniform(-0.22, 0.22) + 0.13 * np.sin(3 * x)
        radius = np.sqrt(
            (X / rng.uniform(0.30, 0.54)) ** 2
            + (Y / rng.uniform(0.30, 0.54)) ** 2
            + (Z / rng.uniform(0.52, 0.76)) ** 2
        )
        inside = radius < 1
        subregion = (Z + 0.24 * Y) > 0.07
        mask = inside * (1 + subregion)
        fn = 0.25 + rng.uniform(0.3, 0.6) * gaussian_filter(inside.astype(float), 0.8)
        fn += 0.06 * np.sin(5 * y) + 0.06 * np.cos(6 * z) + rng.normal(0, 0.07, SHAPE)
        fn += 0.14 * gaussian_filter((mask == 2).astype(float), 1.15)
        volumes.append(np.clip(fn, 0, 1))
        masks.append(mask)
    return (
        torch.tensor(np.asarray(volumes), dtype=torch.float32)[:, None],
        torch.tensor(np.asarray(masks), dtype=torch.float32)[:, None],
        list(range(len(volumes))),
    )


class Warp(nn.Module):
    def __init__(self, shape):
        super().__init__()
        grids = torch.meshgrid(
            *(torch.linspace(-1, 1, n) for n in shape), indexing="ij"
        )
        self.register_buffer("identity", torch.stack(grids[::-1], dim=-1)[None])

    def forward(self, x, flow, mode="bilinear"):
        return F.grid_sample(
            x,
            self.identity + flow.permute(0, 2, 3, 4, 1),
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )


def layer(incoming, width, stride=1):
    return nn.Sequential(
        nn.Conv3d(incoming, width, 3, padding=1, stride=stride, bias=False),
        nn.GroupNorm(4, width),
        nn.LeakyReLU(0.15),
    )


class Register(nn.Module):
    def __init__(self, integrated):
        super().__init__()
        self.integrated = integrated
        self.warp = Warp(SHAPE)
        self.first = nn.Sequential(layer(2, 12), layer(12, 12))
        self.second = nn.Sequential(layer(12, 24, 2), layer(24, 24))
        self.third = nn.Sequential(layer(24, 40, 2), layer(40, 40))
        self.dec2 = nn.Sequential(layer(40 + 24, 24), layer(24, 24))
        self.dec1 = nn.Sequential(layer(24 + 12, 16), layer(16, 16))
        self.head = nn.Conv3d(16, 3, 3, padding=1)
        nn.init.normal_(self.head.weight, 0, 1e-5)
        nn.init.zeros_(self.head.bias)

    def forward(self, moving, fixed):
        first = self.first(torch.cat([moving, fixed], 1))
        second = self.second(first)
        latent = self.third(second)
        x = F.interpolate(
            latent, size=second.shape[-3:], mode="trilinear", align_corners=True
        )
        x = self.dec2(torch.cat([x, second], 1))
        x = F.interpolate(
            x, size=first.shape[-3:], mode="trilinear", align_corners=True
        )
        velocity = self.head(self.dec1(torch.cat([x, first], 1))).tanh() * 0.30
        flow = velocity
        if self.integrated:
            flow = velocity / 16
            for _ in range(4):
                flow = flow + self.warp(flow, flow)
        return self.warp(moving, flow), flow, velocity


def local_corr(x, y):
    def p(z):
        return F.avg_pool3d(z, 5, 1, 2, count_include_pad=False)

    a, b = p(x), p(y)
    covariance = p(x * y) - a * b
    variance = (p(x.square()) - a.square()).clamp_min(0) * (
        p(y.square()) - b.square()
    ).clamp_min(0)
    return (covariance / (variance + 1e-5).sqrt()).mean()


def folds(flow):
    unit = torch.tensor([(n - 1) / 2 for n in SHAPE[::-1]], device=flow.device)
    displacement = flow * unit[None, :, None, None, None]
    jacobian = torch.stack(torch.gradient(displacement, dim=(4, 3, 2)), dim=-1).movedim(
        1, -2
    )
    return float(
        (torch.linalg.det(jacobian + torch.eye(3, device=flow.device)) <= 0)
        .float()
        .mean()
    )


def sample_pairs(ids, count, rng):
    a = rng.integers(0, len(ids), count)
    b = (a + rng.integers(1, len(ids), count)) % len(ids)
    return np.column_stack((ids[a], ids[b]))


def report(model, images, labels, pairs, batch, device):
    if model is not None:
        model.eval()
    all_dice, ncc, fold_rates, sizes = [], [], [], []
    with torch.inference_mode():
        for block in np.array_split(pairs, max(1, int(np.ceil(len(pairs) / batch)))):
            a, b = block.T
            moving, fixed = images[a].to(device), images[b].to(device)
            target, truth = labels[a].to(device), labels[b].to(device)
            if model is None:
                moved, mask, rate = moving, target, 0.0
            else:
                moved, flow, _ = model(moving, fixed)
                mask = model.warp(target, flow, mode="nearest")
                rate = folds(flow)
            pair_dice = []
            for class_id in (1, 2):
                first, second = mask == class_id, truth == class_id
                numerator = 2 * (first & second).sum((1, 2, 3, 4)) + 1e-5
                denominator = first.sum((1, 2, 3, 4)) + second.sum((1, 2, 3, 4)) + 1e-5
                pair_dice.append((numerator / denominator).cpu().numpy())
            all_dice.append(np.stack(pair_dice, axis=1))
            ncc.append(float(local_corr(moved, fixed)))
            fold_rates.append(rate)
            sizes.append(len(block))
    dice = np.concatenate(all_dice)
    return {
        "mean_dice": float(dice.mean()),
        "dice_by_class": dice.mean(axis=0).tolist(),
        "local_ncc": float(np.average(ncc, weights=sizes)),
        "nonpositive_jacobian_fraction": float(np.average(fold_rates, weights=sizes)),
    }


def train(model, images, labels, train_ids, val, test, args, device, output, tag):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    rng = np.random.default_rng(args.seed + 13)
    history, best = [], (-float("inf"), 0)
    path = output / f"{tag}.pt"
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        train_pairs = sample_pairs(train_ids, args.pairs_per_epoch, rng)
        for block in np.array_split(
            train_pairs, int(np.ceil(len(train_pairs) / args.batch_size))
        ):
            a, b = block.T
            moving, fixed = images[a].to(device), images[b].to(device)
            moved, _, velocity = model(moving, fixed)
            smooth = sum(
                (velocity.diff(dim=axis) * 8).square().mean() for axis in (2, 3, 4)
            )
            loss = 1 - local_corr(moved, fixed) + 0.16 * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        val_scores = report(model, images, labels, val, args.batch_size, device)
        history.append(val_scores)
        criterion = (
            val_scores["local_ncc"] - 2 * val_scores["nonpositive_jacobian_fraction"]
        )
        if criterion > best[0]:
            best = (criterion, epoch)
            torch.save(model.state_dict(), path)
        entry = {
            "epoch": epoch,
            "validation": val_scores,
            "seconds": time.monotonic() - start,
        }
        print(tag, entry, flush=True)
        (output / f"{tag}-progress.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch - best[1] >= 4:
            break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "validation": history,
        "selected_epoch": best[1],
        "test": report(model, images, labels, test, args.batch_size, device),
    }


def main(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    images, masks, names = load_mri() if args.real else phantoms(args.seed)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(images))
    a, b = int(0.60 * len(order)), int(0.80 * len(order))
    train_ids, val_ids, test_ids = order[:a], order[a:b], order[b:]
    val, test = sample_pairs(val_ids, 120, rng), sample_pairs(test_ids, 240, rng)
    output = ROOT / (
        "real-data-projects/balakrishnan-hippocampus-registration"
        if args.real
        else "professor-aligned/balakrishnan-deformation-integration"
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "splits.npz",
        source=np.asarray(names),
        train=train_ids,
        validation=val,
        test=test,
    )
    metrics = {
        "identity": {"test": report(None, images, masks, test, args.batch_size, device)}
    }
    for tag, integrated in (("direct", False), ("velocity", True)):
        torch.manual_seed(args.seed)
        model = Register(integrated).to(device, memory_format=torch.channels_last_3d)
        metrics[tag] = train(
            model, images, masks, train_ids, val, test, args, device, output, tag
        )
    outcome = {
        "configuration": vars(args),
        "volumes": len(images),
        "volume_shape": list(SHAPE),
        "validation_pairs": len(val),
        "test_pairs": len(test),
        "metrics": metrics,
    }
    (output / "results.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--seed", type=int, default=56)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--pairs-per-epoch", type=int, default=640)
    main(parser.parse_args())
