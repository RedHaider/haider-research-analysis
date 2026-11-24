# Kvasir Robust Segmentation Benchmark

Real-data companion to the professor-aligned robust medical-segmentation project.
It trains and compares U-Net and DeepLabV3+ encoders on the official Kvasir-SEG
polyp dataset, then evaluates clean accuracy, corruption robustness, calibration,
boundary quality, and inference efficiency.

## Full benchmark design

- Official 1,000-image Kvasir-SEG release; raw data remain outside Git.
- Deterministic five-fold splits with fold manifests saved before training.
- Baseline: ImageNet-pretrained ResNet34 U-Net.
- Proposed: EfficientNet-B0 DeepLabV3+ with corruption augmentation.
- BCE + Dice loss, mixed precision, early stopping and best-checkpoint retention.
- Clean and noise/blur/contrast/JPEG stress tests.
- Dice, IoU, precision, recall, boundary F1, ECE, parameters and latency.
- Fold-level confidence intervals and paired comparisons.

## Completed validated pilot

The completed Colab run used one deterministic held-out split: 680 training,
120 validation and 200 test images at 256 x 256 resolution, with a 20-epoch
maximum and patience of five. All 1,000 masks were validated as non-empty,
lossless binary masks before training.

| Model | Dice | IoU | Precision | Recall | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline U-Net | 0.8427 | 0.7282 | 0.8883 | 0.8016 | 0.0289 |
| Proposed corruption-aware DeepLabV3+ | **0.8714** | **0.7720** | **0.8936** | **0.8502** | **0.0233** |

The proposed model improved Dice by 0.0286 and IoU by 0.0438, increased
recall by 0.0486, and reduced ECE by 0.0056 on this split. These are pilot
results only; they are not five-fold estimates and are not clinical claims.
The machine-readable record is in `validated_fast_report.json`.

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

