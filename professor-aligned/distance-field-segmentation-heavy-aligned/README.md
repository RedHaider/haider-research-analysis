# Anatomical contour learning through distance fields — Professor-aligned simulation

Regress a signed distance transform of the binary vessel mask. Threshold the predicted field for mask and boundary Dice. The held-out test uses the fixed DRIVE partition.

For this run the source observations and labels are simulated.
Input shape is 1 × 64 × 64. There are 160 observations before splitting.

[Research connection](https://usm.maine.edu/directories/people/xin-zhang/)

## Mean held-out metrics

| Arm | dice | boundary_dice | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 0.29777 | 0.26805 | 44608 | 2.0288 | 8 | 8 |
| heavy | 0.8287 | 0.75683 | 1.5488e+06 | 24.155 | 12 | 12 |

Run `python ../../training-logs/batch100/train.py 19` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
