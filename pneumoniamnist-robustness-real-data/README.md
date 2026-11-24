# PneumoniaMNIST Real-Data Robustness Benchmark

Real-data companion to the professor-aligned robust medical-segmentation work for
Truong-Son Hy. The benchmark compares a standard residual CNN baseline with a
corruption-aware version trained using photometric/noise augmentation, label
smoothing, balanced sampling, early stopping and temperature calibration.

The public PneumoniaMNIST dataset is downloaded directly by `medmnist`; raw data,
checkpoints and environments stay outside Git. Evaluation covers the official test
split under clean, Gaussian-noise, blur, contrast and brightness conditions using
AUROC, AUPRC, F1, balanced accuracy and expected calibration error.

## Reproduce

Open `PneumoniaMNIST_Robustness_Colab.ipynb` in Colab and run all cells, or run
`run_benchmark.py` locally. The benchmark performs three seeded runs for each
method and writes a machine-readable `validated_results.json` report.

## Validated results

The completed CPU run used the official train/validation/test splits, three
random seeds, early stopping, and five test conditions. Mean AUROC across the
four corruptions increased from **0.8008 to 0.9232** (**+0.1224**). Gaussian-noise
AUROC increased from **0.6666 to 0.9260**, while clean AUROC remained essentially
unchanged (**0.9412 to 0.9423**). Clean expected calibration error fell from
**0.1261 to 0.0650**. Clean F1 decreased from 0.8746 to 0.8173, so the result is
presented as a robustness/calibration improvement rather than a universal gain.

This is a portfolio research benchmark, not a clinical system and not a
reproduction of Professor Hy's published model.

