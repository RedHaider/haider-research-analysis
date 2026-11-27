# Resolution-gated image inference — Public-data companion

Train a compact CNN on low-resolution versions and a larger CNN on the original input. Choose a confidence threshold on validation data to route uncertain examples to the larger model. This reports the routing fraction and measures no device-level energy or latency.

This run uses the public source linked below.
Input shape is 3 × 32 × 32. There are 1,450 observations before splitting.

[Research connection](https://www.cs.purdue.edu/people/faculty/yunglu.html)

[Dataset source](https://medmnist.com/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran | large_fraction | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.71875 | 0.80774 | 0.7286 | 0.17571 | 31874 | 3.1153 | 5 | 8 | — | — |
| gated | 0.71625 | 0.80552 | 0.72708 | 0.17882 | — | — | — | — | 0.20875 | 0.6 |
| heavy | 0.7 | 0.78955 | 0.70185 | 0.18806 | 1.5253e+06 | 19.46 | 4 | 8 | — | — |

Run `python ../../training-logs/batch100/train.py 9 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
