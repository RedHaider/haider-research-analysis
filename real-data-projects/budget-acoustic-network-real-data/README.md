# Tam Vu — public-data companion

Compare a compact spectrogram CNN with an eight-convolution CNN and evaluate a confidence-routed cascade.

Mini Speech Commands, four spoken commands. The recordings are held out by speaker ID. The compact CNN can accept a confident prediction or defer to the deeper CNN. The threshold is calibrated on validation speakers. This measures conditional routing on public speech, with no earable power measurements.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://www.colorado.edu/cs/tam-vu)

[Dataset source](https://www.tensorflow.org/tutorials/audio/simple_audio)

## Mean held-out metrics

| Model | accuracy | parameters | selected_epoch | medium_fraction | threshold |
|---|---:|---:|---:|---:|---:|
| compact | 0.76171 | 2872 | 7 | — | — |
| medium | 0.942 | 2.6076e+05 | 7 | — | — |
| cascade | 0.93207 | — | — | 0.76693 | 0.73333 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
