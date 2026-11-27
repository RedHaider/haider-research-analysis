# Sleep and wake from wearable heart rate — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www.mtu.edu/data-science/about/faculty/hembroff/)

Input shape is 3 × 10. The experiment uses 3,870 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.95507 | 0.99204 | 0.98622 | 0.033281 | 2236 | 18770 | 7 | 8 |
| heavy | 0.96776 | 0.99695 | 0.99389 | 0.022512 | 2236 | 1.2564e+06 | 3.5 | 7.5 |

Run `python ../../training-logs/batch79/train.py 8` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
