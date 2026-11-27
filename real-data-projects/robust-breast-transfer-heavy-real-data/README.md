# Classification under synthetic acquisition shift — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Classify breast ultrasound images and repeat the test after synthetic blur, gain and additive noise. The benchmark tests robustness to artificial perturbation within a dataset.

[Research connection](https://cs.mst.edu/people/faculty-directory/)

[Dataset source](https://medmnist.com/)

Input shape is 1 × 32 × 32. The experiment uses 702 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.79167 | 0.83281 | 0.92561 | 0.14548 | 425 | 31362 | 5.5 | 8 |
| compact_stress | 0.73077 | 0.7499 | 0.88084 | 0.17499 | — | — | — | — |
| heavy | 0.76923 | 0.78404 | 0.90428 | 0.17348 | 425 | 1.524e+06 | 3.5 | 7.5 |
| heavy_stress | 0.71474 | 0.72598 | 0.8682 | 0.2 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 15 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
