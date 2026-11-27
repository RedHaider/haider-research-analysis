# Readmission prediction with missing prior utilization — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://cancer.osu.edu/find-a-researcher/search-researcher-directory/ping-zhang-100003433)

Input shape is 15. The experiment uses 2,600 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.74164 | 0.79624 | 0.69046 | 0.17641 | 1560 | 3362 | 3 | 7 |
| compact_stress | 0.70819 | 0.77217 | 0.65797 | 0.1862 | — | — | — | — |
| heavy | 0.73495 | 0.79197 | 0.68773 | 0.17834 | 1560 | 6.7328e+05 | 1.5 | 5.5 |
| heavy_stress | 0.71488 | 0.76557 | 0.66615 | 0.18731 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 0` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
