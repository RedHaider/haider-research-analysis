from collections import Counter
import importlib.metadata as deps
import json
import math
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
records = json.loads((root / "training-logs/training-batch-100.json").read_text())
assert len(records) == 40 and len({r["folder"] for r in records}) == 40
assert set(Counter(r["prof"] for r in records).values()) == {2}
keys = [
    "auroc",
    "auroc",
    "auroc",
    "auroc",
    "auroc",
    "localization_mae_px",
    "mae",
    "auroc",
    "elapsed_seconds",
    "accuracy",
    "auroc",
    "auroc",
    "psnr",
    "dice",
    "landmark_error_px",
    "dice",
    "auroc",
    "auroc",
    "psnr",
    "dice",
]
protocol = [
    "Predict whether an ICU stay lasts at least seven days from the first 24 hours of laboratory values and recorded demographics. For stress evaluation, zero four lab channels and their observation flags. The larger model trains with the same masking on one third of each shuffled batch.",
    "Train neural classifiers for heart-disease presence with one entire hospital held out. Report the effect of zeroing three measurement columns at evaluation. Occlusion sensitivity is a feature audit and has no causal interpretation.",
    "Combine image intensity with high-frequency residual texture. Train compact and residual CNNs to classify breast ultrasound images. This is a public ultrasound analogue of computer-aided breast image classification.",
    "Predict 30-day readmission and include an explicit availability indicator beside each input after seeded masking. Split by patient, and report discrimination and calibration for age groups.",
    "Use eight prior irregular glucose readings to predict a recorded value of at least 180 mg/dL at the next eligible time, 0.5 to 8 hours later. Patients remain disjoint across splits.",
    "Recover the translation and rotation applied to a moving 2D image. Use pairs with coordinate channels, hold out original MRI images, and report translation error in pixels and rotation error in degrees. The MRI file does not provide patient identifiers.",
    "Predict the final normalized whole-brain volume from the preceding two values, the prior slope and follow-up history. Each row represents one person and only people with at least three observations qualify.",
    "Classify static versus dynamic activity using accelerometer and gyroscope sequences. Test with a zeroed burst of 16 out of 64 time samples. The larger model also trains with those artificial gaps.",
    "Benchmark training batch sizes 32 and 128 on the same larger MLP, seeds and data split. The elapsed measure includes training and checkpoint selection plus evaluation. Small timing differences can reflect concurrent CPU work.",
    "Train a compact CNN on low-resolution versions and a larger CNN on the original input. Choose a confidence threshold on validation data to route uncertain examples to the larger model. This reports the routing fraction and measures no device-level energy or latency.",
    "Compare a standard classifier with a larger CNN backed by four trainable latent prototypes per class. The prototype head scores cosine similarity. It uses latent vectors rather than named clinical attributes or selected example patches.",
    "Train a temporal CNN to decode left versus right motor imagery. Hold out participants and test with a localized artificial noise burst. The larger model trains with a mix of clean and perturbed examples.",
    "Use sparse, symmetric Fourier lines from already reconstructed 2D MRI images. Train restoration models and score a postprocessing step that enforces agreement with retained Fourier coefficients. This benchmark uses simulated sampling rather than raw scanner k-space.",
    "Compare compact and wider 64-pixel segmentation networks, with clean and perturbed Kvasir-SEG evaluation. The larger arm includes simulated blur and noise during training. Mean Dice and one-pixel boundary Dice are reported.",
    "Regress displacements for four coordinate landmarks between synthetically deformed images. The public companion uses real endoscopy backgrounds, and the displacements are simulated. Derivatives of an original share one split.",
    "Add a high-frequency image channel to thin-vessel segmentation. Train both architectures using foreground-weighted BCE and soft Dice. Use the fixed DRIVE test partition.",
    "Verify radiology image–caption pairs using existing frozen embeddings. Zero a quarter of caption embedding coordinates in the stress condition. The larger verifier trains with a mix of unaltered and altered embeddings.",
    "Use the first six ICU hours to predict a recorded heart rate of at least 115 in hours 6–12. Include stays that have observations in both windows and no early readings above the threshold. Evaluate future observed readings rather than a clinical diagnosis.",
    "Simulate sparse parallel projections and Poisson photon counts from reference CT slices. Train compact and wider image restoration networks against the reference and compare with simulated filtered backprojection. This is slice-level evaluation.",
    "Regress a signed distance transform of the binary vessel mask. Threshold the predicted field for mask and boundary Dice. The held-out test uses the fixed DRIVE partition.",
]
sources = [
    "https://physionet.org/content/challenge-2012/1.0.0/",
    "https://archive.ics.uci.edu/dataset/45/heart+disease",
    "https://medmnist.com/",
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "https://archive.ics.uci.edu/dataset/34/diabetes",
    "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection",
    "https://sites.wustl.edu/oasisbrains/home/oasis-2/",
    "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "https://medmnist.com/",
    "https://medmnist.com/",
    "https://physionet.org/content/eegmmidb/1.0.0/",
    "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection",
    "https://datasets.simula.no/kvasir-seg/",
    "https://datasets.simula.no/kvasir-seg/",
    "https://drive.grand-challenge.org/",
    "https://huggingface.co/datasets/unsloth/Radiology_mini",
    "https://physionet.org/content/challenge-2012/1.0.0/",
    "https://medmnist.com/",
    "https://drive.grand-challenge.org/",
]
overrides = {
    4: "https://ise.washington.edu/facultyfinder/shuai-huang",
    19: "https://usm.maine.edu/directories/people/xin-zhang/",
}
missing = [
    r["folder"] for r in records if not (root / r["folder"] / "results.json").exists()
]
if missing:
    raise SystemExit(f"{len(missing)} pending results: " + ", ".join(missing))
summary = [
    "# Batch 100 — larger neural studies",
    "",
    "Forty trained studies for twenty professors. Both models use two seeds and validation-based model selection. Local CPU runs use up to eight epochs for the compact model and twelve for the larger model, with early stopping. The batch-size study uses the same large architecture and twelve-epoch budget in both arms.",
    "",
    "| Professor | Study | Metric | Baseline arm | Comparison arm | Comparison parameters |",
    "|---|---|---|---:|---:|---:|",
]
for r in records:
    i = r["i"]
    p = root / r["folder"]
    report = json.loads((p / "results.json").read_text())
    assert report["study"]["folder"] == r["folder"] and set(report["runs"]) == {
        "13",
        "37",
    }
    for seed, row in report["runs"].items():
        with np.load(p / "checkpoints" / f"split-{seed}.npz") as f:
            a, b, c = [set(f[k]) for k in ("train", "validation", "test")]
            assert [len(a), len(b), len(c)] == row["sizes"]
            assert not (a & b or b & c or a & c)
        for arm in ("compact", "heavy"):
            model = torch.load(
                p / "checkpoints" / f"{arm}-{seed}.pt",
                weights_only=True,
                map_location="cpu",
            )
            assert model["best_epoch"] == row["metrics"][arm]["best_epoch"]
            assert len(model["validation_loss"]) == row["metrics"][arm]["epochs_ran"]
    means = report["means"]
    for arm, row in means.items():
        for metric, val in row.items():
            expected = (
                sum(s["metrics"][arm][metric] for s in report["runs"].values()) / 2
            )
            assert math.isfinite(val) and math.isclose(
                val, expected, rel_tol=1e-10, abs_tol=1e-10
            )
    label = "Public-data companion" if r["real"] else "Professor-aligned simulation"
    shape = " × ".join(map(str, report["input_shape"]))
    notes = [
        f"# {r['title']} — {label}",
        "",
        protocol[i],
        "",
        (
            "This run uses the public source linked below."
            if r["real"]
            else "For this run the source observations and labels are simulated."
        ),
        f"Input shape is {shape}. There are {report['observations']:,} observations before splitting.",
        "",
        f"[Research connection]({overrides.get(i, r['ref'])})",
    ]
    if r["real"]:
        notes += ["", f"[Dataset source]({sources[i]})"]
    notes += ["", "## Mean held-out metrics", ""]
    cols = list(dict.fromkeys(k for row in means.values() for k in row))
    notes += ["| Arm | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for arm, row in means.items():
        notes += [
            "| "
            + arm
            + " | "
            + " | ".join(f"{row[k]:.5g}" if k in row else "—" for k in cols)
            + " |"
        ]
    notes += [
        "",
        f"Run `python ../../training-logs/batch100/train.py {i}{' --real' if r['real'] else ''}` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.",
        "",
        "See results.json for seed-level metrics and checkpoints for weights and split statistics.",
        "",
    ]
    (p / "README.md").write_text("\n".join(notes), encoding="utf-8")
    requirements = (
        ["numpy", "scipy", "scikit-learn", "Pillow", "torch"]
        + (["openpyxl"] if i == 6 and r["real"] else [])
        + (["edfio"] if i == 11 and r["real"] else [])
    )
    (p / "requirements.txt").write_text(
        "".join(f"{n}=={deps.version(n)}\n" for n in requirements)
    )
    m = keys[i]
    a = means["compact"][m]
    b = means["heavy"][m]
    n = int(means["heavy"]["parameters"])
    summary.append(
        f"| {r['prof']} | [{label}](../{r['folder']}/README.md) | {m} | {a:.4f} | {b:.4f} | {n:,} |"
    )
    if r["real"]:
        print(f"{r['prof']}: {m} {a:.4f} -> {b:.4f}; {n:,} parameters")
(root / "training-logs/batch-100-results.md").write_text(
    "\n".join(summary) + "\n", encoding="utf-8"
)
print("Verified 40 studies and 160 trained checkpoints across 20 professors.")
