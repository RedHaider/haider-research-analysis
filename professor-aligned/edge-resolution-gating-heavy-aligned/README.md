# Resolution-gated image inference — Professor-aligned simulation

Train a compact CNN on low-resolution versions and a larger CNN on the original input. Choose a confidence threshold on validation data to route uncertain examples to the larger model. This reports the routing fraction and measures no device-level energy or latency.

For this run the source observations and labels are simulated.
Input shape is 3 × 32 × 32. There are 900 observations before splitting.

[Research connection](https://www.cs.purdue.edu/people/faculty/yunglu.html)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran | large_fraction | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 4.8726e-06 | 31874 | 2.0157 | 8 | 8 | — | — |
| gated | 1 | 1 | 1 | 4.8726e-06 | — | — | — | — | 0 | 0.6 |
| heavy | 1 | 1 | 1 | 1.3034e-06 | 1.5253e+06 | 11.397 | 3 | 7 | — | — |

Run `python ../../training-logs/batch100/train.py 9` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
