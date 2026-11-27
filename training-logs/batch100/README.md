# Batch 100 training code

The [results index](../batch-100-results.md) links to the 40 studies for 20 professors. Each study has its own README, dependency versions and per-seed results. Simulated studies generate their inputs in Python. The public-data companions use the datasets listed below.

Run from the repository root with the dependencies pinned in the selected study. For example, run the two registration experiments as follows.

```sh
python -m pip install -r real-data-projects/affine-registration-pairs-heavy-real-data/requirements.txt
python training-logs/batch100/train.py 5
python training-logs/batch100/train.py 5 --real
```

The integer selects a study in [the batch manifest](../training-batch-100.json). Training writes checkpoints and updated metrics under the selected study. Set `PORTFOLIO_DATA` to the data directory if it is outside the checkout. The three shared loader modules live under `training-logs/batch50`, `training-logs/batch60` and `training-logs/batch79`.

## Public data paths

Paths are relative to `PORTFOLIO_DATA`, which defaults to the repository’s `datasets` directory. Download the datasets from their publishers and keep them on local storage.

| Path | Source |
|---|---|
| `icu2012/set-a.tar.gz` and `icu2012/Outcomes-a.txt` | [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) |
| `heart-sites/processed.*.data` | [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| `diabetes130.zip` | [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| `diabetes-glucose/Diabetes-Data/data-*` | [UCI Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes) |
| `medmnist/breastmnist_64.npz`, `medmnist/retinamnist.npz` and `medmnist/organamnist.npz` | [Official MedMNIST+ release](https://zenodo.org/records/10519652) |
| `brain-mri-2d.zip` | [Public 2D Brain MRI archive](https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection) |
| `oasis2.xlsx` | [OASIS-2 Longitudinal Subject Data workbook](https://sites.wustl.edu/oasisbrains/home/oasis-2/) |
| `har.zip` | [UCI Human Activity Recognition using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| `eegmmidb/S001R04.edf` through subject 016, runs 04 and 08 | [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/) |
| `kvasir-seg/extracted/Kvasir-SEG/images` and `masks` | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) |
| `drive/train/{input,label}` and `drive/val/{input,label}` | [Pinned DRIVE download and layout](../../steerable-thin-structures-real-data/prepare.py) |
| `roco_clip_embeddings.npz` | [Pinned ROCO-derived feature preparation](../../crossmodal-pair-audit-real-data/prepare.py) |

The published comparison uses the stored per-seed metrics. To rebuild the fixed references after rerunning the relevant experiments locally, run `python training-logs/batch100/fixed_reference.py`. After all 40 reruns, `python training-logs/batch100/verify.py` checks the regenerated checkpoints and metrics.
