# Image-guided marker localization — Public-data companion

Regress the location of an injected soft marker from a single 40×40 image; evaluate centroid error in pixels. Coordinate channels retain position information. Keep all examples made from one background in the same split.

Use real endoscopy backgrounds from Kvasir-SEG. The added marker is synthetic and describes a related image-localization task in a different imaging modality from MRI.

[Research connection](https://sites.google.com/cougarnet.uh.edu/tsekos-site/home)

[Dataset source](https://datasets.simula.no/kvasir-seg/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | mean_error_px | parameters | epoch |
|---|---:|---:|---:|
| compact | 10.09 | 3502 | 5 |
| residual | 7.4234 | 99650 | 5 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 1 --real` from this folder using the dependencies listed in `requirements.txt`.
