# Medical image segmentation with boundary evaluation — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://cse.buffalo.edu/~mgao8/)

## Mean test metrics

| Arm | dice | boundary_dice | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 0.67423 | 0.37873 | 45120 | 8 | 8 |
| heavy | 0.85909 | 0.59601 | 1.5501e+06 | 11.5 | 12 |

Run `python ../../training-logs/batch60/train.py 5` using the pinned dependencies. Public data are read from the project datasets directory.
