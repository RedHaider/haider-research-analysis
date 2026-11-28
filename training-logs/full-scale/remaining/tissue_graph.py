import numpy as np
import torch
from common import DATA, classification, save
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


def region_features(images):
    f = []
    for start in range(0, len(images), 1024):
        data = images[start : start + 1024].astype(np.float32) / 255
        values = []
        for i in range(3):
            for j in range(3):
                p = data[:, i * 9 : (i + 1) * 9, j * 9 : (j + 1) * 9]
                values.append(
                    np.column_stack(
                        [
                            p.mean((1, 2)),
                            p.std((1, 2)),
                            abs(np.diff(p, axis=1)).mean((1, 2)),
                            abs(np.diff(p, axis=2)).mean((1, 2)),
                            np.quantile(p, 0.25, axis=(1, 2)),
                            np.quantile(p, 0.75, axis=(1, 2)),
                        ]
                    )
                )
        f.append(np.stack(values, axis=1).astype(np.float32))
    return np.concatenate(f)


def source(real):
    if real:
        with np.load(DATA / "medmnist/pathmnist.npz", allow_pickle=False) as data:
            return [
                (
                    region_features(data[f"{label}_images"]),
                    data[f"{label}_labels"].ravel().astype(int),
                )
                for label in ("train", "val", "test")
            ]
    r = np.random.default_rng(70)
    y = np.arange(12000) % 9
    latent = r.normal(size=(len(y), 9, 18)).astype(np.float32)
    position = y % 3
    for i in range(9):
        select = position == i % 3
        latent[select, i, :6] += 1.35
    latent[:, :, 6:9] += (y // 3)[:, None, None] * 0.45
    # Separate fixed shuffled splits preserve all class labels.
    index = np.random.default_rng(72).permutation(len(y))
    return [(latent[part], y[part]) for part in np.split(index, [9000, 10500])]


class TileModel(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        sides = np.mgrid[:3, :3].reshape(2, -1).T
        a = (abs(sides[:, None, :] - sides[None, :, :]).sum(2) == 1).astype(np.float32)
        a /= a.sum(1, keepdims=True)
        self.register_buffer("adj", torch.from_numpy(a))
        self.node = nn.Sequential(nn.Linear(18, 128), nn.LayerNorm(128), nn.GELU())
        self.edge = nn.Sequential(
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Linear(160, 96),
            nn.GELU(),
        )
        self.attention = nn.Linear(96, 1)
        self.read = nn.Sequential(
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(192, 9),
        )

    def forward(self, x):
        z = self.node(x)
        h = self.edge(
            torch.cat(
                [z, (self.adj @ z) - z if self.graph else torch.zeros_like(z)], dim=-1
            )
        )
        weights = self.attention(h).softmax(dim=1)
        return self.read(torch.cat([(weights * h).sum(1), h.max(1).values], -1))


def run(real):
    torch.set_num_threads(1)
    (x, y), (v, yy), (t, yt) = source(real)
    scale = StandardScaler().fit(x.reshape(-1, x.shape[-1]))

    def transform(w):
        return torch.from_numpy(
            scale.transform(w.reshape(-1, w.shape[-1]))
            .reshape(w.shape)
            .astype(np.float32)
            .clip(-9, 9)
        )

    train, valid, test = [transform(w) for w in (x, v, t)]
    y_t = torch.from_numpy(y)
    out, results, models = {}, {}, {}
    for name, graph in (("independent_regions", False), ("region_graph", True)):
        torch.manual_seed(70)
        rng = np.random.default_rng(70)
        model = TileModel(graph)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.005)
        best = (-1, 0)
        history = []
        for epoch in range(1, 17):
            model.train()
            order = rng.permutation(len(y))
            for start in range(0, len(y), 384):
                i = order[start : start + 384]
                loss = F.cross_entropy(model(train[i]), y_t[i])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                p = torch.cat([model(b).softmax(1) for b in valid.split(384)]).numpy()
            f = classification(yy, p, np.arange(9))["macro_f1"]
            history.append(f)
            if f > best[0]:
                best = f, epoch
                weights = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                name, "public" if real else "synthetic", epoch, round(f, 4), flush=True
            )
            if epoch - best[1] >= 4:
                break
        model.load_state_dict(weights)
        with torch.inference_mode():
            score = torch.cat([model(b).softmax(1) for b in test.split(384)]).numpy()
        results[name] = {
            **classification(yt, score, np.arange(9)),
            "selected_epoch": best[1],
            "validation_macro_f1": history,
            "parameters": sum(v.numel() for v in model.parameters()),
        }
        models[name] = weights
        out[name] = score
    save(
        "bai-pathmnist-region-graph" if real else "bai-weak-label-region-graph",
        real,
        {
            "target": "whole-patch tissue class; subregion labels are unused",
            "train_tiles": len(y),
            "validation_tiles": len(yy),
            "test_tiles": len(yt),
            "metrics": results,
        },
        {"weights": models, "scaler": scale},
        {"targets": yt, **out},
    )


if __name__ == "__main__":
    for real in (False, True):
        run(real)
