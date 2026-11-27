# Spatial histology information retrieval analogue — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www3.cs.stonybrook.edu/~fuswang/index.html)

Input shape is 5 × 32 × 32. The experiment uses 1,400 observations before splitting.

## Held-out results

| Arm | accuracy | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 1 | 840 | 32729 | 8 | 8 |
| compact_stress | 0.35248 | — | — | — | — |
| heavy | 1 | 840 | 1.5291e+06 | 7 | 9.5 |
| heavy_stress | 0.23137 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 11` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
