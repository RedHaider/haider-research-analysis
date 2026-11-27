# Ruxin (Daisy) Wang — public-data companion

Compare a two-block temporal CNN with an eight-convolution temporal CNN on fused multichannel signals.

PhysioNet EEG Motor Movement/Imagery, 16 people, two runs each, eight scalp channels. Two-second signal windows and participant-disjoint evaluation. The task is left versus right motor imagery.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://scholar.smu.edu/engineering_compsci_etds/51/)

[Dataset source](https://physionet.org/content/eegmmidb/1.0.0/)

## Mean held-out metrics

| Model | auroc | average_precision | accuracy | parameters | selected_epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.53557 | 0.5292 | 0.525 | 1274 | 5 |
| medium | 0.51641 | 0.52434 | 0.51944 | 88130 | 2 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
