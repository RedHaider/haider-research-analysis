# Kvasir Robust Segmentation Benchmark

Real-data companion to the professor-aligned robust medical-segmentation project.
It trains and compares U-Net and DeepLabV3+ encoders on the official Kvasir-SEG
polyp dataset, then evaluates clean accuracy, corruption robustness, calibration,
boundary quality, and inference efficiency.

## Protocol

- Official 1,000-image Kvasir-SEG release; raw data remain outside Git.
- Deterministic five-fold splits with fold manifests saved before training.
- Baseline: ImageNet-pretrained ResNet34 U-Net.
- Proposed: EfficientNet-B0 DeepLabV3+ with corruption augmentation.
- BCE + Dice loss, mixed precision, early stopping and best-checkpoint retention.
- Clean and noise/blur/contrast/JPEG stress tests.
- Dice, IoU, precision, recall, boundary F1, ECE, parameters and latency.
- Fold-level confidence intervals and paired comparisons.

## Data location

The default configuration points to:

`D:/USA based documents/Haider Heavy Research Projects/datasets/kvasir-seg/extracted/Kvasir-SEG`

No medical images, environments or checkpoints are stored on C: or committed to
GitHub.

## Run

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m kvasir_robust.train --config configs/heavy.yaml --fold 0 --model baseline
.venv\Scripts\python -m kvasir_robust.train --config configs/heavy.yaml --fold 0 --model proposed
```

This is a research benchmark, not a clinical system. Kvasir-SEG does not expose
patient identifiers, so the five-fold split is image-level rather than patient-level.
