# Image-guided marker localization — Professor-aligned simulation

Regress the location of an injected soft marker from a single 40×40 image; evaluate centroid error in pixels. Coordinate channels retain position information. Keep all examples made from one background in the same split.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://sites.google.com/cougarnet.uh.edu/tsekos-site/home)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | mean_error_px | parameters | epoch |
|---|---:|---:|---:|
| compact | 9.066 | 3502 | 5.6667 |
| residual | 3.145 | 99650 | 5.6667 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 1` from this folder using the dependencies listed in `requirements.txt`.
