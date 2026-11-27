# Classification under synthetic acquisition shift — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://cs.mst.edu/people/faculty-directory/)

Input shape is 1 × 32 × 32. The experiment uses 1,400 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 9.7631e-06 | 840 | 31362 | 8 | 8 |
| compact_stress | 0.87733 | 1 | 1 | 0.077653 | — | — | — | — |
| heavy | 1 | 1 | 1 | 6.6029e-10 | 840 | 1.524e+06 | 2 | 6 |
| heavy_stress | 0.68323 | 1 | 1 | 0.28833 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 15` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
