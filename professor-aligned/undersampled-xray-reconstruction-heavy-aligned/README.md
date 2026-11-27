# Reconstruction from sparse image measurements — Professor-aligned simulation

Use simulated examples for a controlled benchmark of the central task. Public-data protocols and their populations are documented in the paired study.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://www5.msstate.edu/newsroom/article/2021/04/msus-gurbuz-receives-nsf-career-award-research-advance-smart-sensing)

## Mean test metrics

| Arm | psnr | mae | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 24.079 | 0.047504 | 44864 | 8 | 8 |
| fixed_reference | 25.498 | 0.034484 | — | — | — |
| heavy | 25.9 | 0.03668 | 1.5494e+06 | 12 | 12 |

Run `python ../../training-logs/batch60/train.py 6` using the pinned dependencies. Public data are read from the project datasets directory.
