# Juan (Jen) Li — professor-aligned simulation

Compare a last-reading MLP with a two-layer 96-unit GRU and a regression head.

Generated 90 irregular glucose trajectories with nonlinear periodic dynamics and subject-level baselines.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://pubmed.ncbi.nlm.nih.gov/38672986/)

## Mean held-out metrics

| Model | mae | parameters | selected_epoch |
|---|---:|---:|---:|
| compact | 7.5453 | 145 | 7 |
| medium | 7.4304 | 89953 | 7.6667 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
