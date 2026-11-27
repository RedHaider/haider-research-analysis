# Age-stratified hospital readmission ranking — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://cph.osu.edu/people/fareed5)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.74164 | 0.79624 | 0.69046 | 0.17641 | 3362 | 3 | 7 |
| compact_age_70_plus | 0.70996 | 0.75966 | 0.79844 | 0.20143 | — | — | — |
| compact_under_70 | 0.75462 | 0.79345 | 0.60352 | 0.16613 | — | — | — |
| heavy | 0.73579 | 0.79262 | 0.6865 | 0.17675 | 6.7328e+05 | 1.5 | 5.5 |
| heavy_age_70_plus | 0.68446 | 0.7547 | 0.78883 | 0.20112 | — | — | — |
| heavy_under_70 | 0.75695 | 0.78433 | 0.61016 | 0.16669 | — | — | — |

Run `python ../../training-logs/batch60/train.py 2` using the pinned dependencies. Public data are read from the project datasets directory.
