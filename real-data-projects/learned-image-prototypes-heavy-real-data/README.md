# Prototype-based ultrasound classification — Public-data companion

Compare a standard classifier with a larger CNN backed by four trainable latent prototypes per class. The prototype head scores cosine similarity. It uses latent vectors rather than named clinical attributes or selected example patches.

This run uses the public source linked below.
Input shape is 1 × 64 × 64. There are 702 observations before splitting.

[Research connection](https://umaine.edu/scis/people/chaofan-chen/)

[Dataset source](https://medmnist.com/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.76923 | 0.79177 | 0.89352 | 0.19173 | 31362 | 3.9711 | 4.5 | 7 |
| compact_stress | 0.72115 | 0.70322 | 0.85377 | 0.24629 | — | — | — | — |
| heavy | 0.76923 | 0.82216 | 0.91384 | 0.17113 | 1.4019e+06 | 39.875 | 5.5 | 9.5 |
| heavy_stress | 0.75641 | 0.81892 | 0.91699 | 0.18514 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 10 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
