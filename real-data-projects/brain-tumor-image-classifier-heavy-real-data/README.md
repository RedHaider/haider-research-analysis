# MRI brain tumor image classification — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Classify curated 2D MRI images with and without tumor using a compact CNN and a larger residual CNN. The public archive is deduplicated by file hash. Patient identifiers are unavailable, so this is an image split.

[Research connection](https://www.odu.edu/directory/khan-iftekharuddin)

[Dataset source](https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection)

Input shape is 1 × 64 × 64. The experiment uses 228 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.60377 | 0.75697 | 0.8075 | 0.23462 | 136 | 31362 | 4.5 | 6.5 |
| heavy | 0.4434 | 0.74003 | 0.79786 | 0.2511 | 136 | 1.524e+06 | 1 | 5 |

Run `python ../../training-logs/batch79/train.py 5 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
