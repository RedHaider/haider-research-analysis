# Marios Pattichis — Professor-aligned simulation

Compare one smoothing scale with three in a patch-frequency descriptor combining local residuals and gradient energy. Fit logistic regression and test on original and artificially blurred, darkened and noisy images. The experiment adapts multiscale retinal analysis and uses static images.

This experiment uses generated data with known labels or trajectories.

[Research reference](https://ivpcl.unm.edu/bibtex_php/Journals_Pdfs/2010/Multiscale-AM-FM-Methods-for-Diabetic-Retinopathy-Lesion-Detection.pdf)

## Mean held-out results

| Arm | auroc | average_precision |
|---|---:|---:|
| single_scale_clean | 0.81007 | 0.8218 |
| single_scale_shifted | 0.79469 | 0.80852 |
| multi_scale_clean | 0.8093 | 0.82606 |
| multi_scale_shifted | 0.79393 | 0.80831 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
