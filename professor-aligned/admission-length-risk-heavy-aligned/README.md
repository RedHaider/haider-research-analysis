# Early hospital stay-length classification — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://www.cis.udel.edu/people/faculty/rahmat-beheshti/)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.71154 | 0.76538 | 0.65332 | 0.18858 | 3362 | 4 | 8 |
| heavy | 0.71906 | 0.76456 | 0.66598 | 0.19079 | 6.7328e+05 | 1.5 | 5.5 |

Run `python ../../training-logs/batch60/train.py 4` using the pinned dependencies. Public data are read from the project datasets directory.
