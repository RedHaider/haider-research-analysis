# Denoising microscopy images — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://medicine.osu.edu/find-a-researcher/kimerly-powell-100004501)

## Mean test metrics

| Arm | psnr | mae | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 23.131 | 0.053876 | 45664 | 8 | 8 |
| fixed_reference | 20.273 | 0.077308 | — | — | — |
| heavy | 27.102 | 0.034157 | 1.5514e+06 | 11.5 | 12 |

Run `python ../../training-logs/batch60/train.py 0` using the pinned dependencies. Public data are read from the project datasets directory.
