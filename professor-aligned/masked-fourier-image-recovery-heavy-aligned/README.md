# Image reconstruction from sparse Fourier samples — Professor-aligned simulation

Use sparse, symmetric Fourier lines from already reconstructed 2D MRI images. Train restoration models and score a postprocessing step that enforces agreement with retained Fourier coefficients. This benchmark uses simulated sampling rather than raw scanner k-space.

For this run the source observations and labels are simulated.
Input shape is 2 × 64 × 64. There are 370 observations before splitting.

[Research connection](https://people.cs.rutgers.edu/~dnm/)

## Mean held-out metrics

| Arm | psnr | mae | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 20.229 | 0.071841 | 44864 | 5.4165 | 8 | 8 |
| compact_consistent | 23.849 | 0.050584 | — | — | — | — |
| fixed_reference | 25.014 | 0.043974 | — | — | — | — |
| heavy | 24.354 | 0.047212 | 1.5494e+06 | 39.067 | 12 | 12 |
| heavy_consistent | 25.076 | 0.043797 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 12` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
