# Image reconstruction under simulated motion blur — Public-data companion

Apply a horizontal box blur and additive noise, then compare learned restoration against the blurred input. The public-data companion uses real 64×64 radiographs with synthetic motion blur.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://csiplab.github.io/)

[Data source](https://medmnist.com/)

## Mean test metrics

| Arm | psnr | mae | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 26.295 | 0.037602 | 44608 | 8 | 8 |
| fixed_reference | 26.951 | 0.034466 | — | — | — |
| heavy | 27.789 | 0.031653 | 1.5488e+06 | 8 | 11.5 |

Run `python ../../training-logs/batch60/train.py 7 --real` using the pinned dependencies. Public data are read from the project datasets directory.
