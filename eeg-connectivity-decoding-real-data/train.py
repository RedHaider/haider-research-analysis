"""Nested subject-held-out decoding from spectral power and EEG connectivity."""

import argparse
import hashlib
import json
from pathlib import Path
import edfio
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from download import DATA

ROOT = Path(__file__).resolve().parent
CHANNELS = "fc5 fc3 fc1 fcz fc2 fc4 fc6 c5 c3 c1 cz c2 c4 c6 cp5 cp3 cp1 cpz cp2 cp4 cp6".split()
BANDS = [(8, 13), (13, 20), (20, 30)]


def extract(path):
    f = edfio.read_edf(path)
    fs = f.signals[0].sampling_frequency
    assert fs == 160 and all(s.sampling_frequency == fs for s in f.signals)
    lookup = {s.label.strip(".").lower(): s.data for s in f.signals}
    # Common average reference across all 64 electrodes.
    signals = np.stack([lookup[k] for k in CHANNELS]) - np.stack(
        list(lookup.values())
    ).mean(0)
    filters = [
        sosfiltfilt(butter(4, band, btype="bandpass", fs=fs, output="sos"), signals)
        for band in BANDS
    ]
    upper = np.triu_indices(len(CHANNELS), 1)
    rows = []
    for event in f.annotations:
        if event.text not in ("T1", "T2"):
            continue
        assert event.duration >= 3.5
        start = round((event.onset + 0.5) * fs)
        power = []
        edges = []
        for band in filters:
            epoch = band[:, start : start + int(3 * fs)]
            assert epoch.shape[1] == 480
            power.extend(np.log(np.maximum(epoch.var(1), 1e-12)))
            edges.extend(np.arctanh(np.clip(np.corrcoef(epoch)[upper], -0.995, 0.995)))
        rows.append((power, edges, int(event.text == "T2")))
    return rows


def model(c):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=800, class_weight="balanced")
    )


def scores(y, p):
    return {
        "auroc": roc_auc_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p >= 0.5),
    }


def run(data):
    power = []
    edges = []
    labels = []
    subjects = []
    files = {}
    for subject in range(1, 17):
        for recording in (4, 8):
            path = data / f"S{subject:03d}R{recording:02d}.edf"
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for p, e, y in extract(path):
                power.append(p)
                edges.append(e)
                labels.append(y)
                subjects.append(subject)
    y, groups = np.array(labels), np.array(subjects)
    methods = {
        "power": np.array(power),
        "edges": np.array(edges),
        "fusion": np.c_[power, edges],
    }
    folds = []
    individual = {}
    metrics = {}
    for name, x in methods.items():
        preds = np.full(len(y), np.nan)
        for fold, (train, test) in enumerate(GroupKFold(4).split(x, y, groups)):
            assert not set(groups[train]) & set(groups[test])
            inner = list(GroupKFold(3).split(x[train], y[train], groups[train]))
            candidates = []
            for c in (0.001, 0.01, 0.1, 1.0):
                values = []
                for a, b in inner:
                    fit = model(c).fit(x[train[a]], y[train[a]])
                    values.append(
                        roc_auc_score(y[train[b]], fit.predict_proba(x[train[b]])[:, 1])
                    )
                candidates.append((np.mean(values), c))
            _, c = max(candidates, key=lambda v: v[0])
            fit = model(c).fit(x[train], y[train])
            dest = ROOT / "checkpoints"
            dest.mkdir(exist_ok=True)
            scaler, lr = fit.steps[0][1], fit.steps[1][1]
            np.savez_compressed(
                dest / f"{name}-fold-{fold}.npz",
                mean=scaler.mean_,
                scale=scaler.scale_,
                coef=lr.coef_,
                intercept=lr.intercept_,
                held_out=np.unique(groups[test]),
            )
            preds[test] = fit.predict_proba(x[test])[:, 1]
            folds.append(
                {
                    "method": name,
                    "fold": fold,
                    "C": c,
                    "held_out": np.unique(groups[test]).tolist(),
                    **scores(y[test], preds[test]),
                }
            )
        assert np.isfinite(preds).all()
        np.savez_compressed(
            ROOT / "checkpoints" / f"{name}-predictions.npz",
            subject=groups,
            y=y,
            probability=preds,
        )
        individual[name] = {
            int(g): scores(y[groups == g], preds[groups == g])
            for g in np.unique(groups)
        }
        metrics[name] = {
            k: np.mean([s[k] for s in individual[name].values()])
            for k in ("auroc", "balanced_accuracy")
        }
    diff = np.array(
        [
            individual["fusion"][g]["auroc"] - individual["power"][g]["auroc"]
            for g in individual["power"]
        ]
    )
    boot = (
        np.random.default_rng(17).choice(diff, (10000, len(diff)), replace=True).mean(1)
    )
    return {
        "subjects": 16,
        "epochs": len(y),
        "right_fist_epochs": int(y.sum()),
        "folds": folds,
        "subject_metrics": individual,
        "subject_macro": metrics,
        "fusion_minus_power_auroc_subject_bootstrap_95": np.quantile(
            boot, [0.025, 0.975]
        ).tolist(),
        "edf_sha256": files,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", type=Path, default=DATA)
    (ROOT / "results.json").write_text(json.dumps(run(cli.parse_args().data), indent=2))
