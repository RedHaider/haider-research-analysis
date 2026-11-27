# Geometry-aware landmark displacement estimation — Professor-aligned simulation

Regress displacements for four coordinate landmarks between synthetically deformed images. The public companion uses real endoscopy backgrounds, and the displacements are simulated. Derivatives of an original share one split.

For this run the source observations and labels are simulated.
Input shape is 4 × 64 × 64. There are 960 observations before splitting.

[Research connection](https://personal.utdallas.edu/~xguo/)

## Mean held-out metrics

| Arm | landmark_error_px | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 1.0391 | 32424 | 3.9666 | 8 | 8 |
| heavy | 0.79681 | 1.5281e+06 | 59.669 | 11 | 12 |
| fixed_reference | 3.4928 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 14` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
