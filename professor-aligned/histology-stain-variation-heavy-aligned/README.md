# Histology tissue recognition under stain variation — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://unclineberger.org/people/daiwei-david-zhang/)

Input shape is 3 × 36 × 36. The experiment uses 1,400 observations before splitting.

## Held-out results

| Arm | accuracy | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 1 | 840 | 32217 | 7.5 | 8 |
| compact_stress | 0.73913 | — | — | — | — |
| heavy | 1 | 840 | 1.5278e+06 | 8 | 11.5 |
| heavy_stress | 0.87578 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 3` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
