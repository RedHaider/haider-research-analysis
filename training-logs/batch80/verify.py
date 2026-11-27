from pathlib import Path
import importlib.metadata
import json
import math
import torch

root = Path(__file__).resolve().parents[2]
variants = [("professor-aligned", "aligned"), ("real-data-projects", "real-data")]
index = [
    "# Courosh Mehanian — contrastive pretraining for ultrasound",
    "",
    "Two studies on local CPU with seeds 13 and 37. The classification model receives 64×64 images.",
    "",
    "| Study | Compact AUROC | Larger supervised AUROC | Larger contrastive AUROC |",
    "|---|---:|---:|---:|",
]
for parent, suffix in variants:
    sub = f"{parent}/ultrasound-contrastive-transfer-heavy-{suffix}"
    folder = root / sub
    raw = json.loads((folder / "results.json").read_text())
    real = suffix == "real-data"
    assert set(raw["runs"]) == {"13", "37"}
    expected = [546, 78, 156] if real else [550, 100, 150]
    assert raw["samples"] == expected, raw["samples"]
    for method, mean in raw["means"].items():
        for seed, run in raw["runs"].items():
            ckpt = torch.load(
                folder / "checkpoints" / f"{method}-{seed}.pt",
                weights_only=True,
                map_location="cpu",
            )
            assert ckpt["best_epoch"] == run[method]["best_epoch"]
            assert len(ckpt["pretrain_loss"]) == (
                6 if method == "heavy_contrastive" else 0
            )
            assert all(
                math.isfinite(float(x)) for x in ckpt["finetune_validation_loss"]
            )
        assert all(
            math.isclose(v, sum(r[method][k] for r in raw["runs"].values()) / 2)
            for k, v in mean.items()
        )
    title = "Public-data companion" if real else "Professor-aligned simulation"
    docs = [
        "# " + title,
        "",
        "Test whether contrastive initialization improves training on 64×64 ultrasound images. Train a compact supervised CNN, a larger supervised CNN, and the same larger CNN after six epochs of label-free paired-view contrastive pretraining.",
        "Pretraining uses only training images. The two larger models have the same architecture and each receives ten supervised epochs. Select checkpoints using the separate validation set. Evaluate original images and a fixed, artificial speckle-like perturbation.",
        "",
        "This study adapts ultrasound representation learning to static image classification. The public task uses BreastMNIST breast ultrasound and separates malignant from normal or benign images. The simulation uses echo phantoms with known lesion shapes.",
        "",
        f"Training, validation and test sizes are {expected[0]}, {expected[1]} and {expected[2]}.",
        "",
        "[Research context](https://arxiv.org/abs/2310.10689)",
    ]
    if real:
        docs += [
            "",
            "[Official BreastMNIST+ release](https://zenodo.org/records/10519652)",
        ]
    docs += [
        "",
        "## Evaluation",
        "",
        "| Model | AUROC | Perturbed AUROC | Parameters |",
        "|---|---:|---:|---:|",
    ]
    for arm, row in raw["means"].items():
        docs.append(
            f"| {arm} | {row['auroc']:.4f} | {row['stress_auroc']:.4f} | {int(row['parameters']):,} |"
        )
    docs += [
        "",
        f"Run `python ../../training-logs/batch80/train.py{' --real' if real else ''}` from this folder.",
        "See results.json for per-seed metrics and checkpoints for trained weights, normalization statistics and training curves.",
        "",
    ]
    (folder / "README.md").write_text("\n".join(docs), encoding="utf-8")
    (folder / "requirements.txt").write_text(
        "".join(
            f"{name}=={importlib.metadata.version(name)}\n"
            for name in ["numpy", "scipy", "scikit-learn", "torch", "Pillow"]
        )
    )
    vals = [
        raw["means"][mode]["auroc"]
        for mode in ["compact", "heavy_supervised", "heavy_contrastive"]
    ]
    index.append(
        f"| [{title}](../{sub}/README.md) | "
        + " | ".join(f"{value:.4f}" for value in vals)
        + " |"
    )
    print(title, raw["samples"], json.dumps(raw["means"]))
(root / "training-logs/batch-80-results.md").write_text(
    "\n".join(index) + "\n", encoding="utf-8"
)
