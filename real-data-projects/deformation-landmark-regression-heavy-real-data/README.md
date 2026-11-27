# Geometry-aware landmark displacement estimation — Public-data companion

Regress displacements for four coordinate landmarks between synthetically deformed images. The public companion uses real endoscopy backgrounds, and the displacements are simulated. Derivatives of an original share one split.

This run uses the public source linked below.
Input shape is 4 × 64 × 64. There are 750 observations before splitting.

[Research connection](https://personal.utdallas.edu/~xguo/)

[Dataset source](https://datasets.simula.no/kvasir-seg/)

## Mean held-out metrics

| Arm | landmark_error_px | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 1.2434 | 32424 | 3.3443 | 7 | 8 |
| heavy | 0.94647 | 1.5281e+06 | 46.498 | 11 | 12 |
| fixed_reference | 3.5051 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 14 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
