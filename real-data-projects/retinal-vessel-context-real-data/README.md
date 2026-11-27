# Segmentation of thin vessels — Public-data companion

Segment thin structures using a shallow encoder–decoder and a residual variant with a larger receptive field. Use a foreground-weighted BCE and soft Dice objective. The test set contains unseen images.

Use 40 DRIVE retinal images at 96×96, including 20 held-out masks. This is an anatomical segmentation analogue related to biomedical image segmentation.

[Research connection](https://www.sci.utah.edu/people/tolga.html)

[Dataset source](https://drive.grand-challenge.org/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | dice | parameters | epoch |
|---|---:|---:|---:|
| compact | 0.23558 | 5845 | 10 |
| residual | 0.3784 | 1.0992e+05 | 9.3333 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 8 --real` from this folder using the dependencies listed in `requirements.txt`.
