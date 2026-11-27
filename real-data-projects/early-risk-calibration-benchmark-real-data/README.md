# Aaron Masino — Public-data companion

Train a boosted-tree model on 60% of stays. Fit Platt and isotonic calibration on a disjoint 20%, then score the remaining 20%. Compare risk discrimination, log loss and Brier score. The calibrated models use the same classifier and test set.

PhysioNet Challenge 2012 Set A, 4,000 ICU stays. Use vitals through hour 24 and predict in-hospital mortality. This is a public-data analogue of early clinical risk estimation using a different outcome and patient population from neonatal sepsis.

[Research reference](https://doi.org/10.1371/journal.pone.0212665)

## Mean held-out results

| Arm | auroc | average_precision | brier | log_loss | test_stays |
|---|---:|---:|---:|---:|---:|
| raw | 0.67627 | 0.24997 | 0.11881 | 0.40567 | 800 |
| platt | 0.67627 | 0.24997 | 0.11418 | 0.38033 | 800 |
| isotonic | 0.67545 | 0.2239 | 0.11478 | 0.40957 | 800 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
