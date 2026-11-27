# Anatomical contour learning through distance fields — Public-data companion

Regress a signed distance transform of the binary vessel mask. Threshold the predicted field for mask and boundary Dice. The held-out test uses the fixed DRIVE partition.

This run uses the public source linked below.
Input shape is 1 × 64 × 64. There are 40 observations before splitting.

[Research connection](https://usm.maine.edu/directories/people/xin-zhang/)

[Dataset source](https://drive.grand-challenge.org/)

## Mean held-out metrics

| Arm | dice | boundary_dice | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 0.15461 | 0.15452 | 44608 | 0.366 | 8 | 8 |
| heavy | 0.025695 | 0.025706 | 1.5488e+06 | 2.4897 | 12 | 12 |

Run `python ../../training-logs/batch100/train.py 19 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
