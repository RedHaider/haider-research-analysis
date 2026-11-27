# Confidence and deferral for radiograph classification — Professor-aligned simulation

Train radiograph CNNs and fit a scalar temperature to the validation logits. Report accuracy, AUROC, Brier score and error when retaining the 80% most confident test examples.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://medicine.osu.edu/departments/radiology/our-people/faculty/medical-imaging-informatics)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch | temperature | error_at_80pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.99464 | 0.99994 | 0.021391 | 3286 | 5.3333 | — | — |
| compact_calibrated | 0.99464 | 0.99994 | 0.0063756 | — | — | 0.45 | 0 |
| residual | 1 | 1 | 3.3352e-08 | 99218 | 4.6667 | — | — |
| residual_calibrated | 1 | 1 | 2.4512e-14 | — | — | 0.45 | 0 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 9` from this folder using the dependencies listed in `requirements.txt`.
