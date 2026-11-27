# Clinical readmission prediction with measurement availability — Public-data companion

Predict 30-day readmission and include an explicit availability indicator beside each input after seeded masking. Split by patient, and report discrimination and calibration for age groups.

This run uses the public source linked below.
Input shape is 26. There are 4,200 observations before splitting.

[Research connection](https://bime.uw.edu/faculty/)

[Dataset source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.89149 | 0.60244 | 0.18823 | 0.095052 | 4066 | 0.55882 | 4 | 7.5 |
| compact_age_70_plus | 0.87489 | 0.58712 | 0.21638 | 0.10699 | — | — | — | — |
| compact_under_70 | 0.90546 | 0.61275 | 0.18539 | 0.084999 | — | — | — | — |
| heavy | 0.89097 | 0.60235 | 0.18841 | 0.095446 | 6.8173e+05 | 2.9006 | 2.5 | 6.5 |
| heavy_age_70_plus | 0.87489 | 0.603 | 0.21751 | 0.10729 | — | — | — | — |
| heavy_under_70 | 0.90451 | 0.59867 | 0.18582 | 0.085481 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 3 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
