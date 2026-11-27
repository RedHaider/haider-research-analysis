# Affine registration of image pairs — Professor-aligned simulation

Recover the translation and rotation applied to a moving 2D image. Use pairs with coordinate channels, hold out original MRI images, and report translation error in pixels and rotation error in degrees. The MRI file does not provide patient identifiers.

For this run the source observations and labels are simulated.
Input shape is 4 × 64 × 64. There are 1,110 observations before splitting.

[Research connection](https://cseweb.ucsd.edu/~mniethammer/)

## Mean held-out metrics

| Arm | localization_mae_px | rotation_mae_deg | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 0.90881 | 0.9347 | 32179 | 5.9268 | 5.5 | 8 |
| heavy | 0.67036 | 0.71507 | 1.5263e+06 | 72.399 | 8 | 11.5 |
| fixed_reference | 2.9617 | 4.4867 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 5` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
