# Juan (Jen) Li — public-data companion

Compare a last-reading MLP with a two-layer 96-unit GRU and a regression head.

UCI Diabetes glucose logs. Predict the next recorded glucose from the previous eight readings, observation intervals and time of day. Eligible next readings follow 0.5–12 hours later. Entire participants are held out. This is forecasting, with no treatment recommendation or simulated intervention.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://pubmed.ncbi.nlm.nih.gov/38672986/)

[Dataset source](https://archive.ics.uci.edu/dataset/34/diabetes)

## Mean held-out metrics

| Model | mae | parameters | selected_epoch |
|---|---:|---:|---:|
| compact | 63.863 | 145 | 5.6667 |
| medium | 62.201 | 89953 | 8 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
