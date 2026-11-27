# Denoising microscopy images — Public-data companion

Predict clean 64×64 microscopy crops from images with controlled blur and noise. Compare compact and wide residual encoder–decoders and the corrupted input. The real-data companion groups images by the leading specimen identifier.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://medicine.osu.edu/find-a-researcher/kimerly-powell-100004501)

[Data source](https://data.lhncbc.nlm.nih.gov/public/Malaria/)

## Mean test metrics

| Arm | psnr | mae | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 19.872 | 0.051327 | 45664 | 8 | 8 |
| fixed_reference | 20.378 | 0.071318 | — | — | — |
| heavy | 24.21 | 0.031785 | 1.5514e+06 | 11 | 12 |

Run `python ../../training-logs/batch60/train.py 0 --real` using the pinned dependencies. Public data are read from the project datasets directory.
