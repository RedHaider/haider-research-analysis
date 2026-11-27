# Activity recognition under simulated sensor drift — Public-data companion

Train six-class temporal activity recognition with accelerometer and gyroscope windows. Score the same models on original signals and signals with simulated sensor gain drift and additive noise. Keep participants disjoint, using the official partitions on UCI HAR.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://github.com/WSU-CASAS/ArWISE)

[Data source](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Mean test metrics

| Arm | accuracy | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|
| compact | 0.631 | 19206 | 6 | 8 |
| compact_stress | 0.5175 | — | — | — |
| heavy | 0.709 | 1.2584e+06 | 7.5 | 10 |
| heavy_stress | 0.5365 | — | — | — |

Run `python ../../training-logs/batch60/train.py 9 --real` using the pinned dependencies. Public data are read from the project datasets directory.
