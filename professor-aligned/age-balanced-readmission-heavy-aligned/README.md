# Group reweighting in readmission prediction — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://sail.uark.edu/)

Input shape is 15. The experiment uses 2,600 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.74164 | 0.79624 | 0.69046 | 0.17641 | 1560 | 3362 | 3 | 7 |
| compact_age_70_plus | 0.70996 | 0.75966 | 0.79844 | 0.20143 | — | — | — | — |
| compact_under_70 | 0.75462 | 0.79345 | 0.60352 | 0.16613 | — | — | — | — |
| heavy | 0.7199 | 0.76846 | 0.6643 | 0.18561 | 1560 | 6.7328e+05 | 1 | 5 |
| heavy_age_70_plus | 0.67548 | 0.76484 | 0.80524 | 0.21548 | — | — | — | — |
| heavy_under_70 | 0.73811 | 0.7698 | 0.59553 | 0.17329 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 6` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
