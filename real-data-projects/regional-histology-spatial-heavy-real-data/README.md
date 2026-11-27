# Spatial histology information retrieval analogue — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Train a nine-class tissue classifier with explicit normalized position channels. Evaluate intact image layout and after swapping upper and lower halves. This is a spatial image benchmark related to the professor's spatial medical informatics work.

[Research connection](https://www3.cs.stonybrook.edu/~fuswang/index.html)

[Dataset source](https://medmnist.com/)

Input shape is 5 × 32 × 32. The experiment uses 1,860 observations before splitting.

## Held-out results

| Arm | accuracy | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 0.50761 | 1092 | 32729 | 7.5 | 8 |
| compact_stress | 0.51522 | — | — | — | — |
| heavy | 0.46957 | 1092 | 1.5291e+06 | 4 | 8 |
| heavy_stress | 0.46522 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 11 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
