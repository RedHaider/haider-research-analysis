# Observer model for known image degradation — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://medicine.arizona.edu/radiology/basic-research)

Input shape is 1 × 64 × 64. The experiment uses 900 observations before splitting.

## Held-out results

| Arm | accuracy | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 1 | 540 | 31411 | 8 | 8 |
| heavy | 0.99758 | 540 | 1.5244e+06 | 12 | 12 |

Run `python ../../training-logs/batch79/train.py 17` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
