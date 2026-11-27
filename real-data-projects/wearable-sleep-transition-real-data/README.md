# Transition detection from wearable heart rate — Public-data companion

Predict whether the next 30-second epoch crosses between sleep and wake, using ten past wearable heart-rate epochs and a missing-measurement indicator. The simulated transition label comes from consecutive thresholded synthetic heart-rate samples.

Use heart-rate samples from public wearable recordings and polysomnography sleep/wake labels. The source is a sleep study, with no ICU data.

[Research connection](https://www.nature.com/articles/s41598-019-44004-w)

[Dataset source](https://physionet.org/content/sleep-accel/1.0.0/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.80565 | 0.56909 | 0.18065 | 1846 | 5.3333 |
| residual | 0.74013 | 0.56497 | 0.22941 | 37010 | 1.6667 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 4 --real` from this folder using the dependencies listed in `requirements.txt`.
