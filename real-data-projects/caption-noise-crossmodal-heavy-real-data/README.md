# Imageâ€“text matching with degraded captions — Public-data companion

Verify radiology image–caption pairs using existing frozen embeddings. Zero a quarter of caption embedding coordinates in the stress condition. The larger verifier trains with a mix of unaltered and altered embeddings.

This run uses the public source linked below.
Input shape is 2048. There are 1,120 observations before splitting.

[Research connection](https://engineering.uark.edu/electrical-engineering-computer-science/)

[Dataset source](https://huggingface.co/datasets/unsloth/Radiology_mini)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.76406 | 0.73992 | 0.4511 | 0.16596 | 1.3347e+05 | 0.75481 | 6 | 8 |
| compact_stress | 0.74219 | 0.73854 | 0.44463 | 0.16723 | — | — | — | — |
| fixed_reference | 0.25 | 0.70203 | 0.51523 | 0.34216 | — | — | — | — |
| heavy | 0.77344 | 0.73492 | 0.47692 | 0.16406 | 2.2346e+06 | 5.2901 | 5 | 9 |
| heavy_stress | 0.75937 | 0.73102 | 0.48772 | 0.16557 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 16 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
