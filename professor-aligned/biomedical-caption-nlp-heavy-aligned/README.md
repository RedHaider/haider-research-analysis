# Radiology caption negation detection — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://bime.uw.edu/faculty/meliha-yetisgen/)

Input shape is 1024. The experiment uses 1,100 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.94002 | 0.97908 | 0.97121 | 0.043211 | 679.5 | 67938 | 8 | 8 |
| heavy | 0.94002 | 0.98306 | 0.97634 | 0.039232 | 679.5 | 1.4482e+06 | 6 | 10 |

Run `python ../../training-logs/batch79/train.py 2` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
