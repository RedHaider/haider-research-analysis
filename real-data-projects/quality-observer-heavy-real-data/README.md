# Observer model for known image degradation — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Train a three-class observer to distinguish known levels of synthetic blur and noise on X-rays. This benchmark recovers injected degradation levels rather than radiologist image-quality judgments.

[Research connection](https://medicine.arizona.edu/radiology/basic-research)

[Dataset source](https://medmnist.com/)

Input shape is 1 × 64 × 64. The experiment uses 1,360 observations before splitting.

## Held-out results

| Arm | accuracy | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 0.99674 | 702 | 31411 | 7.5 | 8 |
| heavy | 0.99457 | 702 | 1.5244e+06 | 9 | 11.5 |

Run `python ../../training-logs/batch79/train.py 17 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
