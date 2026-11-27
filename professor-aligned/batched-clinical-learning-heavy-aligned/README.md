# Batch-size trade-offs for large clinical tables — Professor-aligned simulation

Benchmark training batch sizes 32 and 128 on the same larger MLP, seeds and data split. The elapsed measure includes training and checkpoint selection plus evaluation. Small timing differences can reflect concurrent CPU work.

For this run the source observations and labels are simulated.
Input shape is 15. There are 2,600 observations before splitting.

[Research connection](https://www.cs.uaf.edu/faculty-detail/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.73244 | 0.79206 | 0.68529 | 0.1803 | 6.7328e+05 | 2.5944 | 1.5 | 5.5 |
| heavy | 0.72408 | 0.79115 | 0.6788 | 0.18153 | 6.7328e+05 | 1.4115 | 2 | 6 |

Run `python ../../training-logs/batch100/train.py 8` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
