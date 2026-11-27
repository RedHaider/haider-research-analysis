# Longitudinal morphometry forecasting — Professor-aligned simulation

Predict the final normalized whole-brain volume from the preceding two values, the prior slope and follow-up history. Each row represents one person and only people with at least three observations qualify.

For this run the source observations and labels are simulated.
Input shape is 5. There are 340 observations before splitting.

[Research connection](https://unc-public.courseleaf.com/graduate/schools-departments/computer-science/)

## Mean held-out metrics

| Arm | mae | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 0.01754 | 2689 | 0.40805 | 7.5 | 8 |
| heavy | 0.018179 | 6.6509e+05 | 1.2646 | 9 | 11 |
| fixed_reference | 0.024445 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 6` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
