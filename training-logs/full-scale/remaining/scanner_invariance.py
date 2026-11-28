import csv
import io
from zipfile import ZipFile

import numpy as np
import torch
from common import DATA, classification, save, split
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F


def measured():
    x, y, g, device = [], [], [], []
    with ZipFile(DATA / "busbra.zip") as z:
        rows = csv.DictReader(io.TextIOWrapper(z.open("BUSBRA/bus_data.csv")))
        for row in rows:
            with Image.open(z.open(f"BUSBRA/Images/{row['ID']}.png")) as source:
                im = (
                    np.asarray(
                        source.convert("L").resize((32, 32), Image.Resampling.BILINEAR),
                        dtype=np.float32,
                    )
                    / 255
                )
            cells = im.reshape(8, 4, 8, 4)
            x.append(
                np.r_[
                    cells.mean((1, 3)).ravel(),
                    cells.std((1, 3)).ravel(),
                    np.mean(abs(np.diff(im, axis=0)), axis=0),
                    np.mean(abs(np.diff(im, axis=1)), axis=1),
                ]
            )
            y.append(int(row["Pathology"].strip().lower() == "malignant"))
            g.append(row["Case"])
            device.append(row["Device"])
    return np.stack(x), np.asarray(y), np.asarray(g), np.asarray(device)


def simulate():
    r = np.random.default_rng(63)
    n = 11000
    g = np.repeat(np.arange(n // 2), 2)
    scan = np.repeat(np.arange(n // 2) % 4, 2)
    y = r.binomial(1, np.take([0.33, 0.68, 0.75, 0.25], scan))
    x = r.normal(0, 1, (n, 192)) + r.normal(0, 0.4, (n // 2, 192))[g]
    x[:, :35] += y[:, None] * 0.48
    x[:, 35:] += r.normal(0, 1, (4, 157))[scan]
    return x.astype(np.float32), y, g, scan.astype(str)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


class Classifier(nn.Module):
    def __init__(self, width, domains):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(width, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.16),
            nn.Linear(256, 144),
            nn.LayerNorm(144),
            nn.GELU(),
        )
        self.task = nn.Linear(144, 2)
        self.domain = nn.Sequential(
            nn.Linear(144, 96), nn.GELU(), nn.Linear(96, domains)
        )

    def forward(self, x, weight=0):
        z = self.encode(x)
        return self.task(z), self.domain(Reverse.apply(z, weight))


def train_model(x, y, device, tr, valid, test, variant):
    torch.manual_seed(63)
    r = np.random.default_rng(63)
    model = Classifier(x.shape[1], len(torch.unique(device[tr])))
    opt = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.01)
    best = -1
    history = []
    for epoch in range(30):
        model.train()
        for batch in np.array_split(r.permutation(tr), max(1, len(tr) // 128)):
            task, scanner = model(
                x[batch], weight=min(0.45, (epoch + 1) * 0.04) if variant else 0
            )
            loss = F.cross_entropy(task, y[batch]) + F.cross_entropy(
                scanner, device[batch]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            pred, dom = model(x[valid])
            auc = roc_auc_score(y[valid].numpy(), pred.softmax(1)[:, 1].numpy())
            history.append(
                {
                    "validation_auroc": float(auc),
                    "scanner_accuracy": float(
                        (dom.argmax(1) == device[valid]).float().mean()
                    ),
                }
            )
            if auc > best:
                best = auc
                clean = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
    model.load_state_dict(clean)
    model.eval()
    with torch.inference_mode():
        p = model(x[test])[0].softmax(1).numpy()
    return p, clean, history, sum(v.numel() for v in model.parameters())


def main(real):
    torch.set_num_threads(1)
    x, y, g, d = measured() if real else simulate()
    unique = sorted(np.unique(d), key=lambda i: (d == i).sum(), reverse=True)
    target = unique[0]
    gtest = np.unique(g[d == target])
    test = np.flatnonzero(d == target)
    fit = np.flatnonzero(~np.isin(g, gtest) & (d != target))
    assert not set(g[fit]) & set(g[test])
    scope = np.unique(d[fit])
    assert len(scope) >= 2
    codes = np.searchsorted(scope, d)
    counts = np.array([(d[fit] == device).sum() for device in scope])
    assert counts.min() > 0
    train_, val_ = split(y[fit], g[fit], seed=80)
    train, val = fit[train_], fit[val_]
    # Training and validation must both use the same domain coding.
    assert len(np.unique(d[train])) == len(scope)
    scaler = StandardScaler().fit(x[train])
    tx = torch.tensor(scaler.transform(x).clip(-6, 6), dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.int64)
    dd = torch.tensor(codes, dtype=torch.int64)
    metrics, models, outputs = {}, {}, {}
    for name, variant in (("task_only", False), ("scanner_invariant", True)):
        score, weights, history, parameters = train_model(
            tx, yy, dd, train, val, test, variant
        )
        metrics[name] = {
            **classification(y[test], score, np.array([0, 1])),
            "parameters": parameters,
            "validation_history": history,
        }
        models[name] = weights
        outputs[name] = score
    save(
        "moyer-busbra-scanner-shift" if real else "moyer-scanner-invariant-encoding",
        real,
        {
            "held_out_device": str(target),
            "train_images": len(train),
            "validation_images": len(val),
            "test_images": len(test),
            "test_patients": len(np.unique(g[test])),
            "metrics": metrics,
        },
        {"models": models, "scale": scaler, "source_devices": scope},
        {
            "train_groups": g[train],
            "validation_groups": g[val],
            "test_groups": g[test],
            "truth": y[test],
            **outputs,
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
