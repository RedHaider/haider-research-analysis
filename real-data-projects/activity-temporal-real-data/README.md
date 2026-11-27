# Human activity recognition — Public-data companion

Use 64 time samples per window from three accelerometer channels and three gyroscope channels to classify six activities. Participants are disjoint across splits.

Use the official UCI HAR train and test participant partitions. Randomly sample windows within each partition and keep each person grouped.

[Research connection](https://eng.ua.edu/eng-directory/dr-jiaqi-jackey-gong/)

[Dataset source](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | parameters | epoch |
|---|---:|---:|---:|
| compact | 0.61733 | 2118 | 6 |
| residual | 0.65033 | 37614 | 4.6667 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 5 --real` from this folder using the dependencies listed in `requirements.txt`.
