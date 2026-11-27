# Marios Pattichis — Public-data companion

Compare one smoothing scale with three in a patch-frequency descriptor combining local residuals and gradient energy. Fit logistic regression and test on original and artificially blurred, darkened and noisy images. The experiment adapts multiscale retinal analysis and uses static images.

RetinaMNIST fundus images, 1,080 training and 400 test images. Predict grades 2–4 versus grades 0–1. Perturbations are applied to test images to stress the feature representation.

[Research reference](https://ivpcl.unm.edu/bibtex_php/Journals_Pdfs/2010/Multiscale-AM-FM-Methods-for-Diabetic-Retinopathy-Lesion-Detection.pdf)

## Mean held-out results

| Arm | auroc | average_precision |
|---|---:|---:|
| single_scale_clean | 0.73791 | 0.68822 |
| single_scale_shifted | 0.7242 | 0.68362 |
| multi_scale_clean | 0.75315 | 0.694 |
| multi_scale_shifted | 0.72134 | 0.68094 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
