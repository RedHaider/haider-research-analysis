# Anticipating elevated heart rate from early ICU vitals — Public-data companion

Use the first six ICU hours to predict a recorded heart rate of at least 115 in hours 6–12. Include stays that have observations in both windows and no early readings above the threshold. Evaluate future observed readings rather than a clinical diagnosis.

This run uses the public source linked below.
Input shape is 8 × 24. There are 3,045 observations before splitting.

[Research connection](https://engineering.unl.edu/ece/)

[Dataset source](https://physionet.org/content/challenge-2012/1.0.0/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.93295 | 0.72807 | 0.16959 | 0.059711 | 19170 | 3.9836 | 2 | 6 |
| heavy | 0.93295 | 0.71681 | 0.18277 | 0.060086 | 1.2574e+06 | 20.184 | 1.5 | 5.5 |

Run `python ../../training-logs/batch100/train.py 17 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
