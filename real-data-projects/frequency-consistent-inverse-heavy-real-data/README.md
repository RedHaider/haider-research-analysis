# Data-consistent MRI inverse imaging — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Predict a reference MRI image from simulated Fourier undersampling with 16 symmetric sampled lines. Report direct prediction, zero-filled input and a data-consistency projection. The public inputs are 2D magnitude images used as a simulated inverse problem, with no raw complex scanner measurements.

[Research connection](https://cig.ece.wisc.edu/publications/)

[Dataset source](https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection)

Input shape is 2 × 64 × 64. The experiment uses 228 observations before splitting.

## Held-out results

| Arm | psnr | mae | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 16.956 | 0.095802 | 136 | 44864 | 7.5 | 8 |
| compact_consistent | 19.623 | 0.073825 | — | — | — | — |
| fixed_reference | 20.988 | 0.059057 | — | — | — | — |
| heavy | 19.36 | 0.069113 | 136 | 1.5494e+06 | 10.5 | 12 |
| heavy_consistent | 20.851 | 0.061481 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 13 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
