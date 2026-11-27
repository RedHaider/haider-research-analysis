# MRI brain tumor image classification — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www.odu.edu/directory/khan-iftekharuddin)

Input shape is 1 × 64 × 64. The experiment uses 370 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 0.00067243 | 221 | 31362 | 8 | 8 |
| heavy | 1 | 1 | 1 | 3.4133e-06 | 221 | 1.524e+06 | 10 | 12 |

Run `python ../../training-logs/batch79/train.py 5` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
