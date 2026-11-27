# Affine registration of image pairs — Public-data companion

Recover the translation and rotation applied to a moving 2D image. Use pairs with coordinate channels, hold out original MRI images, and report translation error in pixels and rotation error in degrees. The MRI file does not provide patient identifiers.

This run uses the public source linked below.
Input shape is 4 × 64 × 64. There are 684 observations before splitting.

[Research connection](https://cseweb.ucsd.edu/~mniethammer/)

[Dataset source](https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection)

## Mean held-out metrics

| Arm | localization_mae_px | rotation_mae_deg | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 1.1429 | 4.245 | 32179 | 4.237 | 7.5 | 8 |
| heavy | 0.98373 | 4.0761 | 1.5263e+06 | 46.123 | 8 | 11.5 |
| fixed_reference | 3.0564 | 4.4128 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 5 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
