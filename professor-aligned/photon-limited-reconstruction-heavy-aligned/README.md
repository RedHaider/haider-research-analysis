# CT-like reconstruction under simulated count noise — Professor-aligned simulation

Simulate sparse parallel projections and Poisson photon counts from reference CT slices. Train compact and wider image restoration networks against the reference and compare with simulated filtered backprojection. This is slice-level evaluation.

For this run the source observations and labels are simulated.
Input shape is 1 × 32 × 32. There are 440 observations before splitting.

[Research connection](https://eecs.ucmerced.edu/content/changqing-li)

## Mean held-out metrics

| Arm | psnr | mae | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 17.966 | 0.10107 | 44608 | 2.673 | 8 | 8 |
| fixed_reference | 12.951 | 0.18065 | — | — | — | — |
| heavy | 19.081 | 0.088917 | 1.5488e+06 | 16.791 | 11.5 | 12 |

Run `python ../../training-logs/batch100/train.py 18` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
