# Medical image segmentation with boundary evaluation — Public-data companion

Train compact and wide residual encoder–decoders with binary cross-entropy and soft Dice losses. Evaluate mask Dice and one-pixel inner boundary Dice on 64×64 images. Kvasir-SEG splits are at image level.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://cse.buffalo.edu/~mgao8/)

[Data source](https://datasets.simula.no/kvasir-seg/)

## Mean test metrics

| Arm | dice | boundary_dice | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 0.33275 | 0.07552 | 45120 | 5.5 | 8 |
| heavy | 0.41539 | 0.078512 | 1.5501e+06 | 3.5 | 7.5 |

Run `python ../../training-logs/batch60/train.py 5 --real` using the pinned dependencies. Public data are read from the project datasets directory.
