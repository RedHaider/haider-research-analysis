# Artifact-robust biosignal classification — Professor-aligned simulation

Train a temporal CNN to decode left versus right motor imagery. Hold out participants and test with a localized artificial noise burst. The larger model trains with a mix of clean and perturbed examples.

For this run the source observations and labels are simulated.
Input shape is 8 × 128. There are 1,200 observations before splitting.

[Research connection](https://olemiss.edu/profiles/hxiao1)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 4.1639e-08 | 19170 | 3.0499 | 8 | 8 |
| compact_stress | 1 | 1 | 1 | 2.4486e-07 | — | — | — | — |
| heavy | 1 | 1 | 1 | 1.4357e-11 | 1.2574e+06 | 26.976 | 2.5 | 6.5 |
| heavy_stress | 1 | 1 | 1 | 2.7706e-11 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 11` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
