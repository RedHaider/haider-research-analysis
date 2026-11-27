# Kannappan Palaniappan — public-data companion

Compare a compact CNN with an eight-convolution CNN and evaluate a confidence-routed cascade.

NIH Malaria Cell Images. Balance the sampled crops by class. Hold out groups defined by the leading specimen identifier in the filename. Compact CNN confidence can route a crop to the deeper classifier, with a validation-selected threshold. This is an image-level benchmark and does not validate a clinical diagnosis system.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-project.html)

[Dataset source](https://data.lhncbc.nlm.nih.gov/public/Malaria/)

## Mean held-out metrics

| Model | auroc | average_precision | accuracy | parameters | selected_epoch | medium_fraction | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.95479 | 0.91989 | 0.86204 | 3038 | 6.3333 | — | — |
| medium | 0.98615 | 0.98509 | 0.95165 | 2.61e+05 | 6.3333 | — | — |
| cascade | 0.96286 | 0.95046 | 0.93698 | — | — | 0.31886 | 0.8 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
