# Kannappan Palaniappan — professor-aligned simulation

Compare a compact CNN with an eight-convolution CNN and evaluate a confidence-routed cascade.

Generated RGB microscopy-like cell crops with and without inclusions.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-project.html)

## Mean held-out metrics

| Model | auroc | average_precision | accuracy | parameters | selected_epoch | medium_fraction | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.99184 | 0.99236 | 0.95098 | 3038 | 7 | — | — |
| medium | 0.99998 | 0.99998 | 0.99739 | 2.61e+05 | 5 | — | — |
| cascade | 0.99807 | 0.99814 | 0.98954 | — | — | 0.18039 | 0.63333 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
