import importlib.metadata as metadata
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
batch = ROOT / "training-logs/training-batch-50.json"
records = json.loads(batch.read_text(encoding="utf-8"))
assert len(records) == 20 and set(Counter(r["prof"] for r in records).values()) == {2}
metrics = [
    "auroc",
    "mean_error_px",
    "psnr_db",
    "auroc",
    "auroc",
    "accuracy",
    "accuracy",
    "accuracy",
    "dice",
    "brier",
]
protocols = [
    "Binary classification of an observed overnight low-glucose reading using the preceding six readings, observation intervals and time of day. The forecast window is midnight–06:00. Nights require at least one observed outcome-period reading. The estimand is an observed low in an intermittently sampled diary.",
    "Regress the location of an injected soft marker from a single 40×40 image; evaluate centroid error in pixels. Coordinate channels retain position information. Keep all examples made from one background in the same split.",
    "Apply a nine-angle parallel-beam projection and a discretized filtered backprojection. Compare reconstructions against the original slice. Evaluate compact and residual image-to-image CNNs, with the uncorrected backprojection as a reference. Measurements are simulated.",
    "Classify breast-image labels, then repeat evaluation with a fixed synthetic blur, gain change and additive noise. Each architecture sees the same clean training data.",
    "Predict whether the next 30-second epoch crosses between sleep and wake, using ten past wearable heart-rate epochs and a missing-measurement indicator. The simulated transition label comes from consecutive thresholded synthetic heart-rate samples.",
    "Use 64 time samples per window from three accelerometer channels and three gyroscope channels to classify six activities. Participants are disjoint across splits.",
    "Classify nine histology tissue types with compact and residual convolutional classifiers using spatial pooling. Real-data experiments use the official partition and a bounded per-partition sample.",
    "Classify six activities from accelerometer and gyroscope channels, then set gyroscope readings to zero at test time. During residual-model training, mask gyroscope readings for one third of each shuffled batch. The compact baseline trains on intact inputs. Compare both models on intact and missing-stream data. This comparison changes model depth and training augmentation together.",
    "Segment thin structures using a shallow encoder–decoder and a residual variant with a larger receptive field. Use a foreground-weighted BCE and soft Dice objective. The test set contains unseen images.",
    "Train radiograph CNNs and fit a scalar temperature to the validation logits. Report accuracy, AUROC, Brier score and error when retaining the 80% most confident test examples.",
]
sources = [
    "Use the public UCI Diabetes records with glucose dated by patient. The dataset comprises intermittent log entries, with no continuous glucose traces.",
    "Use real endoscopy backgrounds from Kvasir-SEG. The added marker is synthetic and describes a related image-localization task in a different imaging modality from MRI.",
    "Use abdominal CT slices from OrganAMNIST as the reference signal. Simulate sparse projection measurements from the downsampled slices. Evaluation separates slices. Patient identifiers are unavailable in this input.",
    "Use BreastMNIST ultrasound rather than mammograms. The corruption is artificial and does not establish scanner-level or hospital-level robustness.",
    "Use heart-rate samples from public wearable recordings and polysomnography sleep/wake labels. The source is a sleep study, with no ICU data.",
    "Use the official UCI HAR train and test participant partitions. Randomly sample windows within each partition and keep each person grouped.",
    "Use PathMNIST colorectal histology tiles. The official test split is held out and the task is tissue classification.",
    "Use the UCI HAR dual-sensor recordings. A controlled missing stream is introduced at test time.",
    "Use 40 DRIVE retinal images at 96×96, including 20 held-out masks. This is an anatomical segmentation analogue related to biomedical image segmentation.",
    "Use PneumoniaMNIST+ X-rays downsampled to 32×32, with separate official training and test partitions and a validation split drawn from the training set.",
]
index = [
    "# Batch 50 — training and evaluation",
    "",
    "Twenty trained experiments across ten professors. Each study uses seeds 13, 37 and 61, with validation-selected checkpoints.",
    "",
    "| Professor | Study | Primary metric | Compact | Residual | Residual parameters |",
    "|---|---|---|---:|---:|---:|",
]
results = []
for rec in records:
    i = rec["i"]
    p = ROOT / rec["folder"]
    report = json.loads((p / "results.json").read_text())
    assert report["record"]["folder"] == rec["folder"]
    assert set(report["runs"]) == {"13", "37", "61"}
    for seed, r in report["runs"].items():
        with np.load(p / "checkpoints" / f"split-{seed}.npz") as a:
            ids = [set(a[k].tolist()) for k in ("train", "validation", "test")]
            assert [len(s) for s in ids] == r["sizes"]
            assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for arm in ("compact", "residual"):
            state = torch.load(
                p / "checkpoints" / f"{arm}-{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert state["epoch"] == r["metrics"][arm]["epoch"]
    means = report["means"]
    for arm, row in means.items():
        for k, v in row.items():
            values = [r["metrics"][arm][k] for r in report["runs"].values()]
            assert math.isfinite(v) and math.isclose(
                v, sum(values) / len(values), rel_tol=1e-10, abs_tol=1e-10
            )
    key = metrics[i]
    label = "Public-data companion" if rec["real"] else "Professor-aligned simulation"
    description = [
        f"# {rec['title']} — {label}",
        "",
        protocols[i],
        "",
        sources[i]
        if rec["real"]
        else "This benchmark uses synthetic records or images designed for a controlled experiment.",
        "",
        f"[Research connection]({rec['ref']})",
    ]
    if rec["real"]:
        description += ["", f"[Dataset source]({rec['url']})"]
    description += [
        "",
        "Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.",
        "",
        "## Results",
        "",
    ]
    columns = list(dict.fromkeys(k for row in means.values() for k in row))
    description += [
        "| Model | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for arm, row in means.items():
        description.append(
            "| "
            + arm
            + " | "
            + " | ".join(f"{row[k]:.5g}" if k in row else "—" for k in columns)
            + " |"
        )
    command = f"python ../../training-logs/batch50/train.py {i}" + (
        " --real" if rec["real"] else ""
    )
    description += [
        "",
        "The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.",
        "",
        f"Run `{command}` from this folder using the dependencies listed in `requirements.txt`.",
        "",
    ]
    (p / "README.md").write_text("\n".join(description), encoding="utf-8")
    packages = ["numpy", "scipy", "Pillow", "torch", "scikit-learn"]
    (p / "requirements.txt").write_text(
        "".join(f"{name}=={metadata.version(name)}\n" for name in packages)
    )
    index.append(
        f"| {rec['prof']} | [{label}](../{rec['folder']}/README.md) | {key} | {means['compact'][key]:.4f} | {means['residual'][key]:.4f} | {int(means['residual']['parameters']):,} |"
    )
    if rec["real"]:
        results.append((rec["prof"], key, means, report["samples"]))
(ROOT / "training-logs/batch-50-results.md").write_text(
    "\n".join(index) + "\n", encoding="utf-8"
)
print(json.dumps(results, ensure_ascii=True))
