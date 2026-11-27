# Longitudinal morphometry forecasting — Public-data companion

Predict the final normalized whole-brain volume from the preceding two values, the prior slope and follow-up history. Each row represents one person and only people with at least three observations qualify.

This run uses the public source linked below.
Input shape is 5. There are 56 observations before splitting.

[Research connection](https://unc-public.courseleaf.com/graduate/schools-departments/computer-science/)

[Dataset source](https://sites.wustl.edu/oasisbrains/home/oasis-2/)

## Mean held-out metrics

| Arm | mae | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 0.018002 | 2689 | 0.073057 | 8 | 8 |
| heavy | 0.015975 | 6.6509e+05 | 0.32767 | 9.5 | 12 |
| fixed_reference | 0.011146 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 6 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
