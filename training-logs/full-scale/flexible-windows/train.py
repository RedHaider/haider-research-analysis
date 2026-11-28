"""Predict events for multiple observation and future windows on the same timeline."""

import argparse
import csv
import json
import tarfile
from io import TextIOWrapper
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
OBSERVATIONS = (6, 12, 18, 24)
FORECASTS = (4, 8, 12)
VARIABLES = (
    "MAP",
    "NIMAP",
    "HR",
    "RespRate",
    "Temp",
    "SaO2",
    "WBC",
    "Lactate",
    "Creatinine",
    "Urine",
    "Glucose",
    "pH",
    "BUN",
    "GCS",
)
REFERENCE_BP = (0, 1)


def read_cohort(bundle):
    data, ids = [], []
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive:
            identity = Path(member.name).stem
            if not member.isfile() or not identity.isdigit():
                continue
            total = np.zeros((48, len(VARIABLES)), dtype=np.float32)
            count = np.zeros_like(total)
            with TextIOWrapper(archive.extractfile(member)) as stream:
                for row in csv.DictReader(stream):
                    parameter = row["Parameter"]
                    if parameter not in VARIABLES:
                        continue
                    hour = int(row["Time"].split(":")[0])
                    value = float(row["Value"])
                    if hour >= 48 or value <= 0:
                        continue
                    if parameter in ("MAP", "NIMAP") and not 25 <= value <= 180:
                        continue
                    feature = VARIABLES.index(parameter)
                    total[hour, feature] += value
                    count[hour, feature] += 1
            data.append(
                np.divide(
                    total, count, out=np.full_like(total, np.nan), where=count > 0
                )
            )
            ids.append(identity)
    assert len(ids) == 4000, f"Expected 4,000 ICU records in {bundle}"
    return np.stack(data), np.asarray(ids)


def cohort(args):
    if args.real:
        directory = ROOT / "datasets/icu2012"
        arrays, identifiers = zip(
            *(read_cohort(directory / f"set-{part}.tar.gz") for part in "abc")
        )
        ids = np.concatenate(identifiers)
        assert len(np.unique(ids)) == len(ids)
        fit, validation = train_test_split(
            np.arange(4000), test_size=0.20, random_state=args.seed
        )
        return np.concatenate(arrays), ids, (fit, validation, np.arange(4000, 12000))
    rng = np.random.default_rng(args.seed)
    n, t, f = 14000, 48, len(VARIABLES)
    factor = rng.normal(0, 1, (n, 1))
    path = rng.normal(0, 0.24, (n, t)).cumsum(1) + factor
    values = (
        path[..., None] * rng.normal(0, 0.75, (1, 1, f)) + rng.normal(0, 0.6, (n, t, f))
    ).astype(np.float32)
    values[:, :, 0] = 82 - 8.5 * path + rng.normal(0, 6, (n, t))
    values[:, :, 1] = 85 - 8 * path + rng.normal(0, 7, (n, t))
    values[rng.random(values.shape) < 0.55] = np.nan
    ids = np.arange(n).astype(str)
    learn, test = train_test_split(np.arange(n), test_size=0.2, random_state=args.seed)
    train, val = train_test_split(learn, test_size=0.20, random_state=args.seed + 1)
    return values, ids, (train, val, test)


def normalize(values, train):
    center = np.nanmedian(values[train], axis=(0, 1)).astype(np.float32)
    spread = np.nanpercentile(values[train], [25, 75], axis=(0, 1))
    scale = np.maximum((spread[1] - spread[0]).astype(np.float32), 1e-3)
    observed = np.isfinite(values).astype(np.float32)
    normalized = np.nan_to_num(np.clip((values - center) / scale, -8, 8))
    age = np.empty_like(observed)
    missing_hours = np.full(observed.shape[::2], 12, dtype=np.float32)
    for hour in range(observed.shape[1]):
        missing_hours = np.where(
            observed[:, hour] > 0, 0, np.minimum(missing_hours + 1, 12)
        )
        age[:, hour] = missing_hours / 12
    return np.concatenate([normalized, observed, age], axis=2), center, scale


def targets(values):
    branch = values[:, :, list(REFERENCE_BP)]
    nearby = np.isfinite(branch).any(2)
    event = (branch < 65).any(2)
    target = np.zeros(
        (len(values), len(OBSERVATIONS), len(FORECASTS)), dtype=np.float32
    )
    valid = np.zeros(target.shape, dtype=bool)
    for i, cutoff in enumerate(OBSERVATIONS):
        for j, length in enumerate(FORECASTS):
            field = slice(cutoff, cutoff + length)
            target[:, i, j] = event[:, field].any(1)
            valid[:, i, j] = nearby[:, field].sum(1) >= (length + 3) // 4
    return target, valid


class SequenceModel(nn.Module):
    def __init__(self, channels, monotone):
        super().__init__()
        self.monotone = monotone
        self.project = nn.Sequential(
            nn.Linear(channels, 96), nn.LayerNorm(96), nn.SiLU()
        )
        self.recurrent = nn.GRU(96, 128, num_layers=2, dropout=0.15, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(96, max(FORECASTS) if monotone else len(FORECASTS)),
        )

    def forward(self, sequence):
        latent, _ = self.recurrent(self.project(sequence[:, : max(OBSERVATIONS)]))
        raw = self.head(latent[:, np.asarray(OBSERVATIONS) - 1])
        if not self.monotone:
            return raw.sigmoid()
        cumulative_log_survival = torch.cumsum(F.logsigmoid(-raw), dim=-1)
        return -torch.expm1(cumulative_log_survival[..., np.asarray(FORECASTS) - 1])


def metrics(probabilities, labels, mask):
    entries = {}
    for i, obs in enumerate(OBSERVATIONS):
        for j, pred in enumerate(FORECASTS):
            kept = mask[:, i, j]
            p, y = probabilities[kept, i, j], labels[kept, i, j]
            if len(np.unique(y)) != 2:
                raise ValueError(f"Both outcome classes are required at {obs}/{pred}")
            entries[f"{obs}h_to_{pred}h"] = {
                "records": int(kept.sum()),
                "prevalence": float(y.mean()),
                "ap": float(average_precision_score(y, p)),
                "auroc": float(roc_auc_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
            }
    return {
        "windows": entries,
        "macro_ap": float(np.mean([r["ap"] for r in entries.values()])),
        "macro_brier": float(np.mean([r["brier"] for r in entries.values()])),
        "downward_risk_fraction": float(
            (np.diff(probabilities, axis=-1) < -1e-7).mean()
        ),
    }


def summaries(sequence, stop):
    x = sequence[:, :stop]
    return np.concatenate(
        [x.mean(1), x.min(1), x.max(1), x[:, -1], x[:, -min(4, stop) :].mean(1)], axis=1
    )


def fit_boost(x, y, valid, train, test, output, seed):
    predictions = np.empty(y[test].shape)
    for i, stop in enumerate(OBSERVATIONS):
        features = summaries(x, stop)
        for j, horizon in enumerate(FORECASTS):
            indexes = train[valid[train, i, j]]
            estimator = HistGradientBoostingClassifier(
                max_iter=85,
                max_leaf_nodes=19,
                learning_rate=0.073,
                l2_regularization=2.0,
                early_stopping=False,
                random_state=seed,
            )
            estimator.fit(features[indexes], y[indexes, i, j])
            predictions[:, i, j] = estimator.predict_proba(features[test])[:, 1]
            joblib.dump(
                estimator, output / f"boost-{stop}-{horizon}.joblib", compress=3
            )
    return predictions


def fit_gru(x, y, valid, subject, args, path, monotone):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, y, valid = map(torch.from_numpy, (x, y, valid))
    train, validation, test = subject
    model = SequenceModel(x.shape[2], monotone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.004)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history, best = [], (float("inf"), 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train)
        for offset in range(0, len(order), 96):
            indexes = order[offset : offset + 96]
            probabilities = model(x[indexes])
            mask = valid[indexes]
            loss = F.binary_cross_entropy(
                probabilities[mask].clamp(1e-6, 1 - 1e-6), y[indexes][mask]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            prob = torch.cat(
                [model(x[part]) for part in np.array_split(validation, 12)]
            )
        brier = float(
            ((prob[valid[validation]] - y[validation][valid[validation]]) ** 2).mean()
        )
        history.append(brier)
        if brier < best[0]:
            best = (brier, epoch)
            torch.save(model.state_dict(), path)
        print(path.stem, epoch, brier, flush=True)
        if epoch - best[1] >= 5:
            break
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    with torch.inference_mode():
        probability = torch.cat(
            [model(x[part]) for part in np.array_split(test, max(1, len(test) // 160))]
        ).numpy()
    return probability, history, best[1]


def main(args):
    torch.set_num_threads(1)
    signals, ids, subject = cohort(args)
    train, validation, test = subject
    use = (
        "real-data-projects/mehak-icu-flexible-horizons"
        if args.real
        else "professor-aligned/mehak-consistent-window-forecast"
    )
    output = ROOT / use
    output.mkdir(parents=True, exist_ok=True)
    x, center, scale = normalize(signals, train)
    y, valid = targets(signals)
    np.savez_compressed(
        output / "splits.npz",
        train=ids[train],
        validation=ids[validation],
        test=ids[test],
        center=center,
        scale=scale,
        features=VARIABLES,
        observed_hours=OBSERVATIONS,
        forecast_hours=FORECASTS,
    )
    predictions = {
        "pooled_boost": fit_boost(x, y, valid, train, test, output, args.seed)
    }
    scores = {
        "pooled_boost": metrics(predictions["pooled_boost"], y[test], valid[test])
    }
    for name, monotone in (("independent_gru", False), ("monotone_hazard_gru", True)):
        probability, history, epoch = fit_gru(
            x, y, valid, subject, args, output / f"{name}.pt", monotone
        )
        predictions[name] = probability
        scores[name] = {
            "validation_brier": history,
            "selected_epoch": epoch,
            **metrics(probability, y[test], valid[test]),
        }
    np.savez_compressed(
        output / "held-out.npz", labels=y[test], eligibility=valid[test], **predictions
    )
    report = {
        "configuration": vars(args),
        "source_records": len(ids),
        "num_training_records": len(train),
        "num_validation_records": len(validation),
        "num_test_records": len(test),
        "methods": scores,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=46)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
