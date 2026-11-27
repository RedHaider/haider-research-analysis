# Cardiac prediction and feature-occlusion audit — Public-data companion

Train neural classifiers for heart-disease presence with one entire hospital held out. Report the effect of zeroing three measurement columns at evaluation. Occlusion sensitivity is a feature audit and has no causal interpretation.

This run uses the public source linked below.
Input shape is 26. There are 920 observations before splitting.

[Research connection](https://medicine.osu.edu/find-a-researcher/lijun-cheng-100004477)

[Dataset source](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.78061 | 0.87003 | 0.8193 | 0.15623 | 4066 | 0.27599 | 7.5 | 8 |
| compact_stress | 0.69558 | 0.89031 | 0.83969 | 0.19302 | — | — | — | — |
| heavy | 0.66837 | 0.80846 | 0.69661 | 0.22023 | 6.8173e+05 | 1.059 | 2.5 | 6.5 |
| heavy_stress | 0.5068 | 0.8537 | 0.75963 | 0.28117 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 1 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
