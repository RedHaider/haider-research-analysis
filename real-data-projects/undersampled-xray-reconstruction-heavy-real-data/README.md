# Reconstruction from sparse image measurements — Public-data companion

Keep 24% of image pixels using a fixed seeded sampling process. Use normalized local interpolation plus a binary sampling mask as network input. Compare restoration to the interpolated starting image and retain the original 64×64 image as target.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://www5.msstate.edu/newsroom/article/2021/04/msus-gurbuz-receives-nsf-career-award-research-advance-smart-sensing)

[Data source](https://medmnist.com/)

## Mean test metrics

| Arm | psnr | mae | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 25.601 | 0.0404 | 44864 | 7.5 | 8 |
| fixed_reference | 28.634 | 0.022338 | — | — | — |
| heavy | 27.061 | 0.03418 | 1.5494e+06 | 7 | 10.5 |

Run `python ../../training-logs/batch60/train.py 6 --real` using the pinned dependencies. Public data are read from the project datasets directory.
