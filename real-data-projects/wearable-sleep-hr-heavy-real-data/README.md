# Sleep and wake from wearable heart rate — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Predict current sleep or wake from eight past 30-second wearable heart-rate summaries including the current epoch. Train and test on different participants. Public labels come from polysomnography; simulated labels come from thresholded synthetic rates.

[Research connection](https://www.mtu.edu/data-science/about/faculty/hembroff/)

[Dataset source](https://physionet.org/content/sleep-accel/1.0.0/)

Input shape is 3 × 8. The experiment uses 5,336 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.87766 | 0.63879 | 0.91867 | 0.10486 | 3028.5 | 18770 | 3 | 6.5 |
| heavy | 0.87162 | 0.61058 | 0.91087 | 0.11065 | 3028.5 | 1.2564e+06 | 6.5 | 9 |

Run `python ../../training-logs/batch79/train.py 8 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
