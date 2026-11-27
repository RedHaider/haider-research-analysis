# Movement recognition with simulated telemetry gaps — Professor-aligned simulation

Classify static versus dynamic activity using accelerometer and gyroscope sequences. Test with a zeroed burst of 16 out of 64 time samples. The larger model also trains with those artificial gaps.

For this run the source observations and labels are simulated.
Input shape is 6 × 64. There are 1,632 observations before splitting.

[Research connection](https://unc-public.courseleaf.com/graduate/schools-departments/computer-science/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.99632 | 0.99998 | 0.99998 | 0.002866 | 19010 | 3.4421 | 5.5 | 8 |
| compact_stress | 0.86029 | 0.99954 | 0.99959 | 0.10838 | — | — | — | — |
| heavy | 0.99265 | 0.99986 | 0.99987 | 0.0065118 | 1.257e+06 | 31.362 | 7.5 | 9.5 |
| heavy_stress | 0.98407 | 0.99951 | 0.99955 | 0.012358 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 7` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
