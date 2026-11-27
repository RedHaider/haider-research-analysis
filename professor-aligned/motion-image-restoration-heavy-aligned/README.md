# Image reconstruction under simulated motion blur — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://csiplab.github.io/)

## Mean test metrics

| Arm | psnr | mae | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 24.312 | 0.047319 | 44608 | 8 | 8 |
| fixed_reference | 23.921 | 0.049611 | — | — | — |
| heavy | 25.529 | 0.041031 | 1.5488e+06 | 11 | 12 |

Run `python ../../training-logs/batch60/train.py 7` using the pinned dependencies. Public data are read from the project datasets directory.
