# Segmentation robustness to changing image quality — Professor-aligned simulation

Compare compact and wider 64-pixel segmentation networks, with clean and perturbed Kvasir-SEG evaluation. The larger arm includes simulated blur and noise during training. Mean Dice and one-pixel boundary Dice are reported.

For this run the source observations and labels are simulated.
Input shape is 3 × 64 × 64. There are 450 observations before splitting.

[Research connection](https://faculty.ist.psu.edu/suh972/)

## Mean held-out metrics

| Arm | dice | boundary_dice | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 0.74308 | 0.33568 | 45120 | 7.0826 | 8 | 8 |
| compact_stress | 0.06128 | 0.072007 | — | — | — | — |
| heavy | 0.85916 | 0.57092 | 1.5501e+06 | 81.426 | 11 | 12 |
| heavy_stress | 0.85318 | 0.53353 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 13` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
