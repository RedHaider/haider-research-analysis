# Horizon risk prediction with censored follow-up — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Predict an observed death by day 180 using baseline variables and missingness indicators. Exclude records censored before day 180. This complete-outcome benchmark does not estimate an individual survival curve.

[Research connection](https://www.cs.montana.edu/directory/1524592/john-sheppard)

[Dataset source](https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/datasets/lung.csv)

Input shape is 14. The experiment uses 222 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.80769 | 0.78363 | 0.46559 | 0.13757 | 132 | 3298 | 5.5 | 7.5 |
| heavy | 0.75962 | 0.7253 | 0.40946 | 0.18726 | 132 | 6.7251e+05 | 6 | 10 |

Run `python ../../training-logs/batch79/train.py 9 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
