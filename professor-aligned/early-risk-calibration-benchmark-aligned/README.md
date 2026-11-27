# Aaron Masino — Professor-aligned simulation

Train a boosted-tree model on 60% of stays. Fit Platt and isotonic calibration on a disjoint 20%, then score the remaining 20%. Compare risk discrimination, log loss and Brier score. The calibrated models use the same classifier and test set.

This experiment uses generated data with known labels or trajectories.

[Research reference](https://doi.org/10.1371/journal.pone.0212665)

## Mean held-out results

| Arm | auroc | average_precision | brier | log_loss | test_stays |
|---|---:|---:|---:|---:|---:|
| raw | 0.73504 | 0.33678 | 0.1073 | 0.36674 | 800 |
| platt | 0.73504 | 0.33678 | 0.10535 | 0.35281 | 800 |
| isotonic | 0.73393 | 0.30669 | 0.10736 | 0.38517 | 800 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
