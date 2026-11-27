# Longitudinal prediction of elevated glucose — Public-data companion

Use eight prior irregular glucose readings to predict a recorded value of at least 180 mg/dL at the next eligible time, 0.5 to 8 hours later. Patients remain disjoint across splits.

This run uses the public source linked below.
Input shape is 4 × 8. There are 3,797 observations before splitting.

[Research connection](https://ise.washington.edu/facultyfinder/shuai-huang)

[Dataset source](https://archive.ics.uci.edu/dataset/34/diabetes)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.62026 | 0.57992 | 0.45305 | 0.23433 | 18850 | 1.8874 | 1.5 | 5.5 |
| heavy | 0.60512 | 0.58047 | 0.45721 | 0.23799 | 1.2566e+06 | 8.9355 | 1.5 | 5.5 |
| fixed_reference | 0.54753 | 0.5107 | 0.41739 | 0.39701 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 4 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
