# Clinical readmission prediction with measurement availability — Professor-aligned simulation

Predict 30-day readmission and include an explicit availability indicator beside each input after seeded masking. Split by patient, and report discrimination and calibration for age groups.

For this run the source observations and labels are simulated.
Input shape is 30. There are 2,600 observations before splitting.

[Research connection](https://bime.uw.edu/faculty/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.72324 | 0.77809 | 0.67444 | 0.18403 | 4322 | 1.4978 | 5.5 | 8 |
| compact_age_70_plus | 0.684 | 0.75444 | 0.77192 | 0.20591 | — | — | — | — |
| compact_under_70 | 0.73933 | 0.7601 | 0.5875 | 0.175 | — | — | — | — |
| heavy | 0.72492 | 0.78581 | 0.68945 | 0.17931 | 6.848e+05 | 3.7734 | 2 | 6 |
| heavy_age_70_plus | 0.68994 | 0.76933 | 0.79902 | 0.19766 | — | — | — | — |
| heavy_under_70 | 0.73929 | 0.76692 | 0.59534 | 0.17174 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 3` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
