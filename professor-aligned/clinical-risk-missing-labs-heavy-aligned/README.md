# Clinical risk estimation from incomplete lab panels — Professor-aligned simulation

Predict whether an ICU stay lasts at least seven days from the first 24 hours of laboratory values and recorded demographics. For stress evaluation, zero four lab channels and their observation flags. The larger model trains with the same masking on one third of each shuffled batch.

For this run the source observations and labels are simulated.
Input shape is 16 × 24. There are 2,250 observations before splitting.

[Research connection](https://cancer.osu.edu/find-a-researcher/search-researcher-directory/xia-ning-100003296)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.79054 | 0.8222 | 0.70671 | 0.14739 | 19810 | 3.7368 | 2 | 6 |
| compact_stress | 0.72201 | 0.81872 | 0.65524 | 0.18466 | — | — | — | — |
| heavy | 0.78475 | 0.82719 | 0.71064 | 0.14497 | 1.259e+06 | 19.084 | 2 | 6 |
| heavy_stress | 0.76158 | 0.81342 | 0.66951 | 0.16535 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 0` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
