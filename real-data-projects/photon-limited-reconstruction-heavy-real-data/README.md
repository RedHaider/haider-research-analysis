# CT-like reconstruction under simulated count noise — Public-data companion

Simulate sparse parallel projections and Poisson photon counts from reference CT slices. Train compact and wider image restoration networks against the reference and compare with simulated filtered backprojection. This is slice-level evaluation.

This run uses the public source linked below.
Input shape is 1 × 32 × 32. There are 440 observations before splitting.

[Research connection](https://eecs.ucmerced.edu/content/changqing-li)

[Dataset source](https://medmnist.com/)

## Mean held-out metrics

| Arm | psnr | mae | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 14.369 | 0.15394 | 44608 | 2.1018 | 8 | 8 |
| fixed_reference | 10.525 | 0.24706 | — | — | — | — |
| heavy | 15.416 | 0.1331 | 1.5488e+06 | 13.315 | 11.5 | 12 |

Run `python ../../training-logs/batch100/train.py 18 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
