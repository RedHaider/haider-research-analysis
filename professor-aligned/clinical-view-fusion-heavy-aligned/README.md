# Fusion of demographics and early laboratory sequences — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://bime.uw.edu/faculty/chen-liang/)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.79054 | 0.8222 | 0.70671 | 0.14739 | 19810 | 2 | 6 |
| heavy | 0.79633 | 0.83863 | 0.7269 | 0.16171 | 1.259e+06 | 1 | 5 |

Run `python ../../training-logs/batch60/train.py 8` using the pinned dependencies. Public data are read from the project datasets directory.
