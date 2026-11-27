# Forecasting observed hypotension from early ICU vitals — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://profiles.ucsd.edu/shamim.nemati)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.7529 | 0.79167 | 0.64297 | 0.16809 | 19810 | 2 | 6 |
| heavy | 0.73456 | 0.79348 | 0.6345 | 0.17616 | 1.259e+06 | 1 | 5 |

Run `python ../../training-logs/batch60/train.py 3` using the pinned dependencies. Public data are read from the project datasets directory.
