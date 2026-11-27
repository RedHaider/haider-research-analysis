# Tam Vu — professor-aligned simulation

Compare a compact spectrogram CNN with an eight-convolution CNN and evaluate a confidence-routed cascade.

Generated noisy time–frequency patterns in four classes.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://www.colorado.edu/cs/tam-vu)

## Mean held-out metrics

| Model | accuracy | parameters | selected_epoch | medium_fraction | threshold |
|---|---:|---:|---:|---:|---:|
| compact | 0.59638 | 2872 | 7 | — | — |
| medium | 1 | 2.6076e+05 | 8.3333 | — | — |
| cascade | 1 | — | — | 1 | 0.6 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
