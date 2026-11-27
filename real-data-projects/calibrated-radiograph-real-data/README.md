# Confidence and deferral for radiograph classification — Public-data companion

Train radiograph CNNs and fit a scalar temperature to the validation logits. Report accuracy, AUROC, Brier score and error when retaining the 80% most confident test examples.

Use PneumoniaMNIST+ X-rays downsampled to 32×32, with separate official training and test partitions and a validation split drawn from the training set.

[Research connection](https://medicine.osu.edu/departments/radiology/our-people/faculty/medical-imaging-informatics)

[Dataset source](https://medmnist.com/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch | temperature | error_at_80pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.83718 | 0.92106 | 0.1162 | 3286 | 5.3333 | — | — |
| compact_calibrated | 0.83718 | 0.92106 | 0.11965 | — | — | 0.74337 | 0.10497 |
| residual | 0.80256 | 0.95012 | 0.14912 | 99218 | 4.3333 | — | — |
| residual_calibrated | 0.80256 | 0.95012 | 0.14627 | — | — | 1.24 | 0.1226 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 9 --real` from this folder using the dependencies listed in `requirements.txt`.
