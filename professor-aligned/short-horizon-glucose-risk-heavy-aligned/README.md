# Longitudinal prediction of elevated glucose — Professor-aligned simulation

Use eight prior irregular glucose readings to predict a recorded value of at least 180 mg/dL at the next eligible time, 0.5 to 8 hours later. Patients remain disjoint across splits.

For this run the source observations and labels are simulated.
Input shape is 4 × 8. There are 4,550 observations before splitting.

[Research connection](https://ise.washington.edu/facultyfinder/shuai-huang)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.83213 | 0.89844 | 0.80334 | 0.1168 | 18850 | 3.0332 | 4 | 7.5 |
| heavy | 0.8448 | 0.9088 | 0.82128 | 0.11178 | 1.2566e+06 | 12.046 | 2.5 | 6.5 |
| fixed_reference | 0.76606 | 0.8191 | 0.6623 | 0.16522 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 4` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
