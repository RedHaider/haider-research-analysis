# Horizon risk prediction with censored follow-up — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www.cs.montana.edu/directory/1524592/john-sheppard)

Input shape is 14. The experiment uses 950 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.66895 | 0.67944 | 0.5384 | 0.21631 | 570 | 3298 | 8 | 8 |
| heavy | 0.65297 | 0.68728 | 0.54547 | 0.22319 | 570 | 6.7251e+05 | 4 | 8 |

Run `python ../../training-logs/batch79/train.py 9` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
