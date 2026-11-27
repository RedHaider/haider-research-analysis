# Multimodal radiology pair verification — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://midas.umich.edu/directory/chuan-zhou/)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.98672 | 0.99951 | 0.99859 | 0.0084671 | 1.3347e+05 | 3 | 7 |
| fixed_reference | 0.27344 | 0.99993 | 0.99981 | 0.24118 | — | — | — |
| heavy | 0.98516 | 0.99924 | 0.99784 | 0.012175 | 2.2346e+06 | 2 | 6 |

Run `python ../../training-logs/batch60/train.py 1` using the pinned dependencies. Public data are read from the project datasets directory.
