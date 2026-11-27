# Artifact-robust biosignal classification — Public-data companion

Train a temporal CNN to decode left versus right motor imagery. Hold out participants and test with a localized artificial noise burst. The larger model trains with a mix of clean and perturbed examples.

This run uses the public source linked below.
Input shape is 8 × 128. There are 480 observations before splitting.

[Research connection](https://olemiss.edu/profiles/hxiao1)

[Dataset source](https://physionet.org/content/eegmmidb/1.0.0/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.5 | 0.48458 | 0.50651 | 0.25057 | 19170 | 1.1596 | 2 | 6 |
| compact_stress | 0.5 | 0.47513 | 0.51699 | 0.25047 | — | — | — | — |
| heavy | 0.47917 | 0.48736 | 0.4957 | 0.25047 | 1.2574e+06 | 12.525 | 2 | 6 |
| heavy_stress | 0.4875 | 0.51264 | 0.51277 | 0.25022 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 11 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
