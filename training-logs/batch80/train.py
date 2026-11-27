import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "batch79"))
from train import DeepNetwork, scores

SEEDS = (13, 37)
torch.set_num_threads(1)


def dataset(real):
    if real:
        with np.load(ROOT / "datasets/medmnist/breastmnist_64.npz") as a:
            return tuple(
                (
                    a[f"{p}_images"][:, None].astype(np.float32) / 255,
                    1 - a[f"{p}_labels"].ravel().astype(int),
                )
                for p in ("train", "val", "test")
            )
    rng = np.random.default_rng(95)
    u, v = np.mgrid[-1:1:64j, -1:1:64j]
    x = []
    y = np.arange(800) % 2
    rng.shuffle(y)
    for label in y:
        a, b = rng.normal(0, 0.18, 2)
        radius = ((u - a) / (0.25 + 0.04 * label)) ** 2 + (
            (v - b) / (0.26 - 0.045 * label)
        ) ** 2
        echo = 0.35 + 0.08 * np.cos(v * 20) + 0.38 * np.exp(-radius)
        echo -= 0.15 * label * np.exp(-(((u - a) / 0.23) ** 2)) * (v > b + 0.23)
        echo = echo * (1 + rng.normal(0, 0.24, u.shape)) + rng.normal(0, 0.035, u.shape)
        x.append(np.clip(echo, 0, 1))
    x = np.asarray(x, dtype=np.float32)[:, None]
    return (x[:550], y[:550]), (x[550:650], y[550:650]), (x[650:], y[650:])


def transform(x):
    shift = torch.randint(-4, 5, (2,))
    y = torch.roll(x, (int(shift[0]), int(shift[1])), dims=(-2, -1))
    mask = torch.rand(len(y), 1, 1, 1) < 0.5
    y = torch.where(mask, y.flip(-1), y)
    gain = 0.85 + 0.3 * torch.rand((len(y), 1, 1, 1))
    return (y * gain + torch.randn_like(y) * 0.04).contiguous(
        memory_format=torch.channels_last
    )


def encode(model, x):
    return model.context(model.stage(model.stem(x))).mean((2, 3))


def pretrain(model, x):
    projector = nn.Sequential(nn.Linear(352, 128), nn.GELU(), nn.Linear(128, 64))
    # Train the encoder and projector without labels. The classifier head starts untouched.
    opt = torch.optim.AdamW(
        [
            *model.stem.parameters(),
            *model.stage.parameters(),
            *model.context.parameters(),
            *projector.parameters(),
        ],
        lr=0.0014,
        weight_decay=0.002,
    )
    losses = []
    for epoch in range(6):
        model.train()
        projector.train()
        running = []
        for ix in torch.randperm(len(x)).split(48):
            if len(ix) < 2:
                continue
            a = F.normalize(projector(encode(model, transform(x[ix]))), dim=1)
            b = F.normalize(projector(encode(model, transform(x[ix]))), dim=1)
            logits = a @ b.T / 0.13
            match = torch.arange(len(ix))
            loss = (
                F.cross_entropy(logits, match) + F.cross_entropy(logits.T, match)
            ) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())
        losses.append(float(np.mean(running)))
    return losses


def experiment(parts, seed, out):
    (train, y), (valid, v), (test, t) = parts
    mu = train.mean()
    std = train.std() + 1e-5

    def tensor(x):
        return torch.from_numpy((x - mu) / std).contiguous(
            memory_format=torch.channels_last
        )

    a, b, c = map(tensor, (train, valid, test))
    y, v, t = [torch.as_tensor(z, dtype=torch.long) for z in (y, v, t)]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    result = {}
    for method, heavy in [
        ("compact", False),
        ("heavy_supervised", True),
        ("heavy_contrastive", True),
    ]:
        torch.manual_seed(seed)
        model = DeepNetwork((1, 64, 64), 2, False, heavy).to(
            memory_format=torch.channels_last
        )
        pre = pretrain(model, a) if method == "heavy_contrastive" else []
        opt = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.002)
        best = float("inf")
        results = []
        weight = torch.bincount(y, minlength=2).float()
        weight = weight.sum() / (2 * weight.clamp(min=1))
        for epoch in range(10 if heavy else 8):
            model.train()
            for ix in torch.randperm(len(a)).split(48):
                loss = F.cross_entropy(model(transform(a[ix])), y[ix], weight=weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                risk = F.cross_entropy(model(b), v).item()
            results.append(risk)
            if risk < best:
                best = risk
                best_epoch = epoch + 1
                state = {
                    k: value.detach().clone() for k, value in model.state_dict().items()
                }
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            p = model(c).numpy()
            generator = torch.Generator().manual_seed(9904)
            noisy = (
                c + 0.23 * torch.randn(c.shape, generator=generator) * c.abs()
            ).contiguous(memory_format=torch.channels_last)
            stress = model(noisy).numpy()
        result[method] = {
            **scores(t.numpy(), p, "classification"),
            "stress_auroc": scores(t.numpy(), stress, "classification")["auroc"],
            "parameters": sum(value.numel() for value in model.parameters()),
            "best_epoch": best_epoch,
        }
        torch.save(
            {
                "weights": state,
                "best_epoch": best_epoch,
                "finetune_validation_loss": results,
                "pretrain_loss": pre,
                "training_mean": float(mu),
                "training_std": float(std),
            },
            out / "checkpoints" / f"{method}-{seed}.pt",
        )
        print(seed, method, result[method], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    prefix = "real-data-projects" if args.real else "professor-aligned"
    suffix = "real-data" if args.real else "aligned"
    out = ROOT / prefix / f"ultrasound-contrastive-transfer-heavy-{suffix}"
    parts = dataset(args.real)
    runs = {seed: experiment(parts, seed, out) for seed in SEEDS}
    means = {
        k: {m: float(np.mean([r[k][m] for r in runs.values()])) for m in vals}
        for k, vals in runs[13].items()
    }
    (out / "results.json").write_text(
        json.dumps(
            {"samples": [len(y) for _, y in parts], "means": means, "runs": runs},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
