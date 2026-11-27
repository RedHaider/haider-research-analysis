# Prototype-based ultrasound classification — Professor-aligned simulation

Compare a standard classifier with a larger CNN backed by four trainable latent prototypes per class. The prototype head scores cosine similarity. It uses latent vectors rather than named clinical attributes or selected example patches.

For this run the source observations and labels are simulated.
Input shape is 1 × 64 × 64. There are 950 observations before splitting.

[Research connection](https://umaine.edu/scis/people/chaofan-chen/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 1.5192e-06 | 31362 | 4.8632 | 8 | 8 |
| compact_stress | 1 | 1 | 1 | 0.00016947 | — | — | — | — |
| heavy | 1 | 1 | 1 | 8.7564e-08 | 1.4019e+06 | 63.928 | 12 | 12 |
| heavy_stress | 1 | 1 | 1 | 1.7427e-07 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 10` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
