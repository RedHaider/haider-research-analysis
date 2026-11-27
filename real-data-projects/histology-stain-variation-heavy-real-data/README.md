# Histology tissue recognition under stain variation — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Classify nine tissue types and repeat tests with synthetic channel gain and offset changes. The public-data run uses the official PathMNIST partition.

[Research connection](https://unclineberger.org/people/daiwei-david-zhang/)

[Dataset source](https://medmnist.com/)

Input shape is 3 × 36 × 36. The experiment uses 1,860 observations before splitting.

## Held-out results

| Arm | accuracy | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 0.63478 | 1092 | 32217 | 7.5 | 8 |
| compact_stress | 0.15109 | — | — | — | — |
| heavy | 0.60978 | 1092 | 1.5278e+06 | 5 | 9 |
| heavy_stress | 0.13913 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 3 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
