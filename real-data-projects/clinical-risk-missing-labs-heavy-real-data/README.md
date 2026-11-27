# Clinical risk estimation from incomplete lab panels — Public-data companion

Predict whether an ICU stay lasts at least seven days from the first 24 hours of laboratory values and recorded demographics. For stress evaluation, zero four lab channels and their observation flags. The larger model trains with the same masking on one third of each shuffled batch.

This run uses the public source linked below.
Input shape is 18 × 24. There are 4,000 observations before splitting.

[Research connection](https://cancer.osu.edu/find-a-researcher/search-researcher-directory/xia-ning-100003296)

[Dataset source](https://physionet.org/content/challenge-2012/1.0.0/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.71413 | 0.58187 | 0.76306 | 0.20191 | 19970 | 6.1249 | 1.5 | 5.5 |
| compact_stress | 0.70761 | 0.58215 | 0.76197 | 0.20985 | — | — | — | — |
| heavy | 0.71467 | 0.5722 | 0.7563 | 0.20199 | 1.2594e+06 | 25.986 | 1 | 5 |
| heavy_stress | 0.71467 | 0.57069 | 0.74813 | 0.20291 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 0 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
