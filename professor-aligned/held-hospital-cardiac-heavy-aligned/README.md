# Cardiac risk on a held-out hospital — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://medicine.osu.edu/departments/biomedical-informatics/our-people)

Input shape is 26. The experiment uses 1,100 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.71273 | 0.77831 | 0.74398 | 0.19043 | 643 | 4066 | 5.5 | 7.5 |
| heavy | 0.72 | 0.77694 | 0.72671 | 0.19173 | 643 | 6.8173e+05 | 2 | 6 |

Run `python ../../training-logs/batch79/train.py 1` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
