# Movement recognition with simulated telemetry gaps — Public-data companion

Classify static versus dynamic activity using accelerometer and gyroscope sequences. Test with a zeroed burst of 16 out of 64 time samples. The larger model also trains with those artificial gaps.

This run uses the public source linked below.
Input shape is 6 × 64. There are 3,200 observations before splitting.

[Research connection](https://unc-public.courseleaf.com/graduate/schools-departments/computer-science/)

[Dataset source](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.9995 | 0.99996 | 0.99997 | 0.00054252 | 19010 | 7.2249 | 8 | 8 |
| compact_stress | 0.9975 | 0.99997 | 0.99997 | 0.0023191 | — | — | — | — |
| heavy | 0.999 | 0.99997 | 0.99998 | 0.00085224 | 1.257e+06 | 52.676 | 4.5 | 8.5 |
| heavy_stress | 0.9985 | 1 | 1 | 0.0013562 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 7 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
