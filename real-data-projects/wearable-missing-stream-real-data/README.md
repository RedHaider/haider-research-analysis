# Wearable classification with missing sensors — Public-data companion

Classify six activities from accelerometer and gyroscope channels, then set gyroscope readings to zero at test time. During residual-model training, mask gyroscope readings for one third of each shuffled batch. The compact baseline trains on intact inputs. Compare both models on intact and missing-stream data. This comparison changes model depth and training augmentation together.

Use the UCI HAR dual-sensor recordings. A controlled missing stream is introduced at test time.

[Research connection](https://engineering.tamu.edu/cse/profiles/mortazavi-bobak.html)

[Dataset source](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | parameters | epoch |
|---|---:|---:|---:|
| compact | 0.611 | 2118 | 6 |
| compact_stress | 0.53333 | — | — |
| residual | 0.65033 | 37614 | 5.6667 |
| residual_stress | 0.59833 | — | — |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 7 --real` from this folder using the dependencies listed in `requirements.txt`.
