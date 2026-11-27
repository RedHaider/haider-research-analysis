# Batch-size trade-offs for large clinical tables — Public-data companion

Benchmark training batch sizes 32 and 128 on the same larger MLP, seeds and data split. The elapsed measure includes training and checkpoint selection plus evaluation. Small timing differences can reflect concurrent CPU work.

This run uses the public source linked below.
Input shape is 13. There are 4,200 observations before splitting.

[Research connection](https://www.cs.uaf.edu/faculty-detail/)

[Dataset source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.88596 | 0.61696 | 0.22384 | 0.098502 | 6.7175e+05 | 7.03 | 3 | 7 |
| heavy | 0.88596 | 0.64069 | 0.24109 | 0.096977 | 6.7175e+05 | 3.3026 | 2.5 | 6.5 |

Run `python ../../training-logs/batch100/train.py 8 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
