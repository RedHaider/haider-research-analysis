# Data-consistent MRI inverse imaging — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://cig.ece.wisc.edu/publications/)

Input shape is 2 × 64 × 64. The experiment uses 370 observations before splitting.

## Held-out results

| Arm | psnr | mae | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 20.078 | 0.073572 | 221 | 44864 | 8 | 8 |
| compact_consistent | 22.232 | 0.06004 | — | — | — | — |
| fixed_reference | 24.54 | 0.046359 | — | — | — | — |
| heavy | 23.98 | 0.04932 | 221 | 1.5494e+06 | 12 | 12 |
| heavy_consistent | 24.59 | 0.046241 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 13` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
