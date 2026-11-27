# Activity recognition under simulated sensor drift — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://github.com/WSU-CASAS/ArWISE)

## Mean test metrics

| Arm | accuracy | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|
| compact | 0.98652 | 19206 | 7.5 | 8 |
| compact_stress | 0.4473 | — | — | — |
| heavy | 0.97426 | 1.2584e+06 | 4.5 | 8.5 |
| heavy_stress | 0.46936 | — | — | — |

Run `python ../../training-logs/batch60/train.py 9` using the pinned dependencies. Public data are read from the project datasets directory.
