# Ruxin (Daisy) Wang — professor-aligned simulation

Compare a two-block temporal CNN with an eight-convolution temporal CNN on fused multichannel signals.

Generated paired left/right sensor streams with frequency cues and heterogeneous synthetic participants.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://scholar.smu.edu/engineering_compsci_etds/51/)

## Mean held-out metrics

| Model | auroc | average_precision | accuracy | parameters | selected_epoch |
|---|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 1274 | 7 |
| medium | 1 | 1 | 1 | 88130 | 9 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
