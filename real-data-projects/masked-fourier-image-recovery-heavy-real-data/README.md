# Image reconstruction from sparse Fourier samples — Public-data companion

Use sparse, symmetric Fourier lines from already reconstructed 2D MRI images. Train restoration models and score a postprocessing step that enforces agreement with retained Fourier coefficients. This benchmark uses simulated sampling rather than raw scanner k-space.

This run uses the public source linked below.
Input shape is 2 × 64 × 64. There are 228 observations before splitting.

[Research connection](https://people.cs.rutgers.edu/~dnm/)

[Dataset source](https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection)

## Mean held-out metrics

| Arm | psnr | mae | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 16.814 | 0.09689 | 44864 | 3.9425 | 7.5 | 8 |
| compact_consistent | 20.621 | 0.065709 | — | — | — | — |
| fixed_reference | 21.494 | 0.057618 | — | — | — | — |
| heavy | 19.581 | 0.066406 | 1.5494e+06 | 24.28 | 11 | 12 |
| heavy_consistent | 21.418 | 0.059075 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 12 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
