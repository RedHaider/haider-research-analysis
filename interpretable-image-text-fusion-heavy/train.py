"""Train and ablate two-stream fusion under heterogeneous modality quality."""

import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch import nn

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent


def sample(n, seed, maps, severe=False):
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(n, 8))
    logits = hidden[:, 0] + hidden[:, 4] + 0.65 * hidden[:, 1] * hidden[:, 5]
    labels = logits + rng.normal(0, 0.45, n) > 0
    streams = []
    for i in range(2):
        weak = rng.random(n) < (0.55 if severe else 0.22)
        signal = hidden[:, i * 4 : i * 4 + 4] @ maps[i]
        noise = rng.normal(size=signal.shape) * np.where(weak, 2.6, 0.45)[:, None]
        streams.append(signal + noise)
    return torch.tensor(np.stack(streams, axis=1), dtype=torch.float32), torch.tensor(
        labels, dtype=torch.float32
    )


class Fusion(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(32, 48),
                    nn.LayerNorm(48),
                    nn.GELU(),
                    nn.Linear(48, 24),
                    nn.GELU(),
                )
                for _ in range(2)
            ]
        )
        self.gate = nn.Linear(48, 2)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        z = [layer(x[:, i]) for i, layer in enumerate(self.encoders)]
        if self.mode == "image":
            z[1] = torch.zeros_like(z[1])
        if self.mode == "text":
            z[0] = torch.zeros_like(z[0])
        if self.mode == "gated":
            g = self.gate(torch.cat(z, 1)).softmax(1)
            z = [v * g[:, i : i + 1] for i, v in enumerate(z)]
        return self.head(torch.cat(z, 1)).flatten()


def fit(seed, mode, train, val):
    torch.manual_seed(seed)
    model = Fusion(mode)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    best_loss = float("inf")
    for epoch in range(16):
        for ids in torch.randperm(len(train[0])).split(128):
            pred = model(train[0][ids])
            loss = nn.functional.binary_cross_entropy_with_logits(pred, train[1][ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(val[0]), val[1]
            ).item()
        if loss < best_loss:
            best_loss = loss
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model.eval()


def run(seed):
    maps = np.random.default_rng(seed).normal(0, 0.65, (2, 4, 32))
    train, val = [sample(n, seed + j, maps) for n, j in [(4800, 1), (800, 2)]]
    runs = {}
    for mode in ["image", "text", "concat", "gated"]:
        model = fit(seed, mode, train, val)
        dest = ROOT / "checkpoints"
        dest.mkdir(exist_ok=True)
        torch.save(model.state_dict(), dest / f"{mode}-{seed}.pt")
        for split, severe in [("clean", False), ("degraded", True)]:
            x, y = sample(1600, seed + 3, maps, severe)
            with torch.no_grad():
                p = model(x).sigmoid().numpy()
            runs[mode + "_" + split] = {
                "auroc": roc_auc_score(y, p),
                "brier": brier_score_loss(y, p),
            }
    return runs


if __name__ == "__main__":
    runs = {s: run(s) for s in [17, 29, 43]}
    means = {
        k: {m: float(np.mean([r[k][m] for r in runs.values()])) for m in v}
        for k, v in next(iter(runs.values())).items()
    }
    (ROOT / "results.json").write_text(
        json.dumps({"runs": runs, "means": means}, indent=2)
    )
