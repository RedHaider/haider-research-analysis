import importlib.metadata as md
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
manifest = root / "training-logs/training-batch-60.json"
records = json.loads(manifest.read_text(encoding="utf-8"))
assert len(records) == 20 and set(Counter(r["prof"] for r in records).values()) == {2}
metric = [
    "psnr",
    "auroc",
    "auroc",
    "auroc",
    "auroc",
    "dice",
    "psnr",
    "psnr",
    "auroc",
    "accuracy",
]
frameworks = [
    "Predict clean 64×64 microscopy crops from images with controlled blur and noise. Compare compact and wide residual encoder–decoders and the corrupted input. The real-data companion groups images by the leading specimen identifier.",
    "Train compact and wide multilayer perceptrons on frozen image and text embeddings, absolute differences and elementwise products. Select three high-similarity negative captions within the same ten-pair block. Keep blocks disjoint between training and validation and use a separate test gallery. Compare with cosine similarity.",
    "Compare neural readmission rankers, and report AUROC, average precision and Brier score for the full test set and by age below or at least 70. Hold out people. The public-data endpoint is observed readmission within 30 days for diabetes encounters.",
    "Use 24 temporal bins before a fixed landmark. The public-data study uses observations before ICU hour six and predicts an observed systolic reading below 90 mmHg in hours 6–12, conditional on recorded normal prior blood pressure and available follow-up. The endpoint is a recorded low reading.",
    "Test neural prediction of stays of seven or more days. The public-data model uses admission-type and source codes, prior utilization, age and a gender indicator, with patient-disjoint splits.",
    "Train compact and wide residual encoder–decoders with binary cross-entropy and soft Dice losses. Evaluate mask Dice and one-pixel inner boundary Dice on 64×64 images. Kvasir-SEG splits are at image level.",
    "Keep 24% of image pixels using a fixed seeded sampling process. Use normalized local interpolation plus a binary sampling mask as network input. Compare restoration to the interpolated starting image and retain the original 64×64 image as target.",
    "Apply a horizontal box blur and additive noise, then compare learned restoration against the blurred input. The public-data companion uses real 64×64 radiographs with synthetic motion blur.",
    "Combine temporal laboratory values and measurement indicators with age and recorded sex. Predict a stay of at least seven days from the first 24 hours of public ICU records. Report held-out stays and standard binary prediction metrics.",
    "Train six-class temporal activity recognition with accelerometer and gyroscope windows. Score the same models on original signals and signals with simulated sensor gain drift and additive noise. Keep participants disjoint, using the official partitions on UCI HAR.",
]
datasets = [
    "https://data.lhncbc.nlm.nih.gov/public/Malaria/",
    "https://huggingface.co/datasets/unsloth/Radiology_mini",
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "https://physionet.org/content/challenge-2012/1.0.0/",
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "https://datasets.simula.no/kvasir-seg/",
    "https://medmnist.com/",
    "https://medmnist.com/",
    "https://physionet.org/content/challenge-2012/1.0.0/",
    "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
]
overrides = {
    2: "https://cph.osu.edu/people/fareed5",
    5: "https://cse.buffalo.edu/~mgao8/",
    8: "https://bime.uw.edu/faculty/chen-liang/",
    9: "https://github.com/WSU-CASAS/ArWISE",
}
lines = [
    "# Batch 60 — larger neural benchmarks",
    "",
    "Twenty trained studies for ten professors. Both arms use two seeds, local CPU, and validation-selected checkpoints. The larger arm can train up to 12 epochs and the compact arm up to eight.",
    "",
    "| Professor | Study | Metric | Compact | Larger | Larger parameters | Input dimensions |",
    "|---|---|---|---:|---:|---:|---|",
]
results = []
for rec in records:
    i = rec["i"]
    folder = root / rec["folder"]
    report = json.loads((folder / "results.json").read_text())
    assert report["study"]["folder"] == rec["folder"] and set(report["runs"]) == {
        "13",
        "37",
    }
    for seed, run in report["runs"].items():
        with np.load(folder / "checkpoints" / f"split-{seed}.npz") as stats:
            ids = [set(stats[k]) for k in ("train", "validation", "test")]
            assert [len(k) for k in ids] == run["sizes"] and not (
                ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2]
            )
        for arm in ("compact", "heavy"):
            saved = torch.load(
                folder / "checkpoints" / f"{arm}-{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert saved["best_epoch"] == run["metrics"][arm]["best_epoch"]
            assert len(saved["validation_loss"]) == run["metrics"][arm]["epochs_ran"]
    means = report["means"]
    for arm, values in means.items():
        for key, value in values.items():
            expected = sum(r["metrics"][arm][key] for r in report["runs"].values()) / 2
            assert math.isfinite(value) and math.isclose(
                value, expected, rel_tol=1e-10, abs_tol=1e-10
            )
    m = metric[i]
    label = "Public-data companion" if rec["real"] else "Professor-aligned simulation"
    instruction = (
        frameworks[i]
        if rec["real"]
        else "Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study."
    )
    readme = [
        f"# {rec['title']} — {label}",
        "",
        instruction,
        "",
        "Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.",
        "",
        f"[Research reference]({overrides.get(i, rec['ref'])})",
    ]
    if rec["real"]:
        readme += ["", f"[Data source]({datasets[i]})"]
    readme += ["", "## Mean test metrics", ""]
    columns = list(dict.fromkeys(key for row in means.values() for key in row))
    readme += [
        "| Arm | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for arm, vals in means.items():
        readme += [
            "| "
            + arm
            + " | "
            + " | ".join(f"{vals[k]:.5g}" if k in vals else "—" for k in columns)
            + " |"
        ]
    readme += [
        "",
        f"Run `python ../../training-logs/batch60/train.py {i}{' --real' if rec['real'] else ''}` using the pinned dependencies. Public data are read from the project datasets directory.",
        "",
    ]
    (folder / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (folder / "requirements.txt").write_text(
        "".join(
            f"{k}=={md.version(k)}\n"
            for k in ("numpy", "scipy", "scikit-learn", "Pillow", "torch")
        )
    )
    size = int(means["heavy"]["parameters"])
    form = " × ".join(str(a) for a in report["input_shape"])
    lines += [
        f"| {rec['prof']} | [{label}](../{rec['folder']}/README.md) | {m} | {means['compact'][m]:.4f} | {means['heavy'][m]:.4f} | {size:,} | {form} |"
    ]
    if rec["real"]:
        results.append(
            {
                "prof": rec["prof"],
                "metric": m,
                "shape": report["input_shape"],
                "n": report["observations"],
                "means": means,
                "epochs": [
                    v["metrics"]["heavy"]["epochs_ran"] for v in report["runs"].values()
                ],
            }
        )
(root / "training-logs/batch-60-results.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print(json.dumps(results))
