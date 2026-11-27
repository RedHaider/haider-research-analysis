# Segmentation robustness to changing image quality — Public-data companion

Compare compact and wider 64-pixel segmentation networks, with clean and perturbed Kvasir-SEG evaluation. The larger arm includes simulated blur and noise during training. Mean Dice and one-pixel boundary Dice are reported.

This run uses the public source linked below.
Input shape is 3 × 64 × 64. There are 500 observations before splitting.

[Research connection](https://faculty.ist.psu.edu/suh972/)

[Dataset source](https://datasets.simula.no/kvasir-seg/)

## Mean held-out metrics

| Arm | dice | boundary_dice | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 0.40038 | 0.06244 | 45120 | 6.2249 | 4.5 | 8 |
| compact_stress | 0.27809 | 0.053732 | — | — | — | — |
| heavy | 0.43489 | 0.070568 | 1.5501e+06 | 34.118 | 3 | 7 |
| heavy_stress | 0.44646 | 0.079006 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 13 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
