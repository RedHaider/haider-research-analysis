# Cardiac prediction and feature-occlusion audit — Professor-aligned simulation

Train neural classifiers for heart-disease presence with one entire hospital held out. Report the effect of zeroing three measurement columns at evaluation. Occlusion sensitivity is a feature audit and has no causal interpretation.

For this run the source observations and labels are simulated.
Input shape is 26. There are 1,100 observations before splitting.

[Research connection](https://medicine.osu.edu/find-a-researcher/lijun-cheng-100004477)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.71273 | 0.77831 | 0.74398 | 0.19043 | 4066 | 0.6282 | 5.5 | 7.5 |
| compact_stress | 0.66 | 0.71007 | 0.63171 | 0.2174 | — | — | — | — |
| heavy | 0.72 | 0.77694 | 0.72671 | 0.19173 | 6.8173e+05 | 2.1087 | 2 | 6 |
| heavy_stress | 0.62727 | 0.7053 | 0.62219 | 0.22047 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 1` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
