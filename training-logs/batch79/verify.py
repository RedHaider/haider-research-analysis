from collections import Counter
import importlib.metadata as meta
import json
import math
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
records = json.loads((root / "training-logs/training-batch-79.json").read_text())
assert len(records) == 38 and len(Counter(r["prof"] for r in records)) == 19
assert set(Counter(r["prof"] for r in records).values()) == {2}
metric = [
    "auroc",
    "auroc",
    "auroc",
    "accuracy",
    "auroc",
    "auroc",
    "auroc",
    "endpoint_error_px",
    "auroc",
    "auroc",
    "auroc",
    "accuracy",
    "auroc",
    "psnr",
    "auroc",
    "auroc",
    "auroc",
    "accuracy",
    "psnr",
]
methods = [
    "Train on intact utilization features and score both intact data and cases with three history fields zeroed. The larger model sees zeroed history fields in one-third of each shuffled batch. Prediction target is 30-day readmission.",
    "Hold one site out for testing. Fill nonfinite inputs with zero and append missingness indicators. Classification is presence of recorded heart disease.",
    "Detect lexicon-defined negation cues in radiology figure captions after removing the source cue words. Tokenized word n-grams are hashed to 1,024 features. The labels are weak labels, and the task concerns captions rather than clinical notes.",
    "Classify nine tissue types and repeat tests with synthetic channel gain and offset changes. The public-data run uses the official PathMNIST partition.",
    "Predict ICU stays of seven or more days from first-day lab sequences, observation flags and demographics. The larger model trains with a block of lab inputs and flags removed in one-third of each shuffled batch. The same missing-block condition is used during stress evaluation.",
    "Classify curated 2D MRI images with and without tumor using a compact CNN and a larger residual CNN. The public archive is deduplicated by file hash. Patient identifiers are unavailable, so this is an image split.",
    "Compare an unweighted compact model with a wider model trained using inverse frequency weights for combinations of age group and outcome. Report discrimination and Brier score separately by age. Architecture and training weights both change.",
    "Train two dense image-to-image networks to predict two components of a displacement field. Bound the predicted displacement to eight pixels per axis. Apply smooth synthetic deformation to source images and keep derivatives of each original in the same split. The real-data study uses real endoscopy backgrounds and artificial motion.",
    "Predict current sleep or wake from eight past 30-second wearable heart-rate summaries including the current epoch. Train and test on different participants. Public labels come from polysomnography; simulated labels come from thresholded synthetic rates.",
    "Predict an observed death by day 180 using baseline variables and missingness indicators. Exclude records censored before day 180. This complete-outcome benchmark does not estimate an individual survival curve.",
    "Classify left versus right motor imagery using EEG time series. Hold out people and repeat evaluation after additional temporal smoothing. Smoothing approximates reduced sensor bandwidth.",
    "Train a nine-class tissue classifier with explicit normalized position channels. Evaluate intact image layout and after swapping upper and lower halves. This is a spatial image benchmark related to the professor's spatial medical informatics work.",
    "Train a pair verifier on frozen radiology image and caption embeddings. Train the larger verifier with image-embedding noise in one-third of each batch. Test both models with the same intensity of added noise. Training blocks and the test gallery are separated.",
    "Predict a reference MRI image from simulated Fourier undersampling with 16 symmetric sampled lines. Report direct prediction, zero-filled input and a data-consistency projection. The public inputs are 2D magnitude images used as a simulated inverse problem, with no raw complex scanner measurements.",
    "Compare compact and larger readmission classifiers under different training-data budgets. Sample at most 650 training records for the compact arm and at most 3,600 for the wider arm. Patients are held out. This design changes architecture and sample size together.",
    "Classify breast ultrasound images and repeat the test after synthetic blur, gain and additive noise. The benchmark tests robustness to artificial perturbation within a dataset.",
    "Classify RetinaMNIST grades 2–4 versus grades 0–1. Evaluate the official test partition as a related retinal disease classification task.",
    "Train a three-class observer to distinguish known levels of synthetic blur and noise on X-rays. This benchmark recovers injected degradation levels rather than radiologist image-quality judgments.",
    "Restore MRI image crops after synthetic Rician-like magnitude noise. Compare both trained architectures with the corrupted input. The data are 2D images with file-level deduplication and no patient identifiers.",
]
urls = [
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "https://archive.ics.uci.edu/dataset/45/heart+disease",
    "https://huggingface.co/datasets/unsloth/Radiology_mini",
    "https://medmnist.com/",
    "https://physionet.org/content/challenge-2012/1.0.0/",
    "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection",
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "https://datasets.simula.no/kvasir-seg/",
    "https://physionet.org/content/sleep-accel/1.0.0/",
    "https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/datasets/lung.csv",
    "https://physionet.org/content/eegmmidb/1.0.0/",
    "https://medmnist.com/",
    "https://huggingface.co/datasets/unsloth/Radiology_mini",
    "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection",
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "https://medmnist.com/",
    "https://medmnist.com/",
    "https://medmnist.com/",
    "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection",
]
rows = []
summary = [
    "# Batch 79 — larger neural studies",
    "",
    "38 studies across 19 professors. Each result averages seeds 13 and 37. Experiments ran locally on CPU with validation-selected models and kept the original training and evaluation partitions.",
    "",
    "| Professor | Study | Metric | Compact | Larger | Larger parameters | Input dimensions |",
    "|---|---|---|---:|---:|---:|---|",
]
missing = [
    r["folder"] for r in records if not (root / r["folder"] / "results.json").is_file()
]
if missing:
    raise SystemExit(f"{len(missing)} runs pending: " + ", ".join(missing))
for rec in records:
    i = rec["i"]
    p = root / rec["folder"]
    report = json.loads((p / "results.json").read_text())
    assert report["study"]["folder"] == rec["folder"]
    assert set(report["runs"]) == {"13", "37"}
    for seed, run in report["runs"].items():
        with np.load(p / "checkpoints" / f"split-{seed}.npz") as arrays:
            tr, val, te = (set(arrays[k]) for k in ("train", "validation", "test"))
            assert not (tr & val or tr & te or val & te)
            assert [len(a) for a in (tr, val, te)] == run["sizes"]
        for arm in ("compact", "heavy"):
            found = torch.load(
                p / "checkpoints" / f"{arm}-{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert found["best_epoch"] == run["metrics"][arm]["best_epoch"]
            assert len(found["validation_loss"]) == run["metrics"][arm]["epochs_ran"]
    means = report["means"]
    for arm, values in means.items():
        for name, value in values.items():
            expected = sum(r["metrics"][arm][name] for r in report["runs"].values()) / 2
            assert math.isfinite(value) and math.isclose(
                value, expected, abs_tol=1e-10, rel_tol=1e-10
            )
    label = "Public-data companion" if rec["real"] else "Professor-aligned simulation"
    if i == 7:
        assert all(means[arm]["endpoint_error_px"] < 20 for arm in ("compact", "heavy"))
    inp = " × ".join(map(str, report["input_shape"]))
    key = metric[i]
    title = rec["title"]
    content = [
        f"# {title} — {label}",
        "",
        "Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.",
        "",
        methods[i]
        if rec["real"]
        else "Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.",
        "",
        f"[Research connection]({rec['ref']})",
    ]
    if rec["real"]:
        content += ["", f"[Dataset source]({urls[i]})"]
    content += [
        "",
        f"Input shape is {inp}. The experiment uses {report['observations']:,} observations before splitting.",
        "",
        "## Held-out results",
        "",
    ]
    columns = list(dict.fromkeys(c for row in means.values() for c in row))
    content += [
        "| Arm | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for arm, row in means.items():
        content += [
            "| "
            + arm
            + " | "
            + " | ".join(f"{row[k]:.5g}" if k in row else "—" for k in columns)
            + " |"
        ]
    content += [
        "",
        f"Run `python ../../training-logs/batch79/train.py {i}{' --real' if rec['real'] else ''}` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.",
        "",
        "See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.",
        "",
    ]
    (p / "README.md").write_text("\n".join(content), encoding="utf-8")
    deps = ["numpy", "scipy", "scikit-learn", "Pillow", "torch"] + (
        ["edfio"] if rec["real"] and i == 10 else []
    )
    (p / "requirements.txt").write_text(
        "".join(f"{k}=={meta.version(k)}\n" for k in deps)
    )
    n = int(means["heavy"]["parameters"])
    summary.append(
        f"| {rec['prof']} | [{label}](../{rec['folder']}/README.md) | {key} | {means['compact'][key]:.4f} | {means['heavy'][key]:.4f} | {n:,} | {inp} |"
    )
    if rec["real"]:
        rows.append((rec["prof"], key, means, n))
(root / "training-logs/batch-79-results.md").write_text(
    "\n".join(summary) + "\n", encoding="utf-8"
)
print(f"Verified {len(records)} studies for {len(rows)} professors.")
for name, k, rank, n in rows:
    print(
        f"{name}: {k} {rank['compact'][k]:.4f} -> {rank['heavy'][k]:.4f}; {n:,} parameters"
    )
