# Anticipating elevated heart rate from early ICU vitals — Professor-aligned simulation

Use the first six ICU hours to predict a recorded heart rate of at least 115 in hours 6–12. Include stays that have observations in both windows and no early readings above the threshold. Evaluate future observed readings rather than a clinical diagnosis.

For this run the source observations and labels are simulated.
Input shape is 8 × 24. There are 1,800 observations before splitting.

[Research connection](https://engineering.unl.edu/ece/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.75 | 0.47794 | 0.25616 | 0.19776 | 19170 | 2.4072 | 2 | 6 |
| heavy | 0.75 | 0.54716 | 0.28566 | 0.18824 | 1.2574e+06 | 10.924 | 1 | 5 |

Run `python ../../training-logs/batch100/train.py 17` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
