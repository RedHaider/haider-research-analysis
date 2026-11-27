# Trained professor portfolio studies

Each index links to the professor-aligned study and public-data companion for each professor in that batch. The folders contain descriptions, dependency versions, held-out metrics, and either an individual training script or instructions for the shared trainer.

| Batch | Professors in batch | Studies | Results |
|---|---:|---:|---|
| 35 | 5 | 10 | [Pilot comparisons](batch-35-results.md) |
| 40 | 5 | 10 | [Medium neural networks](batch-40-results.md) |
| 50 | 10 | 20 | [Residual neural networks](batch-50-results.md) |
| 60 | 10 | 20 | [Larger neural networks](batch-60-results.md) |
| 79 | 19 | 38 | [Larger neural networks](batch-79-results.md) |
| 80 | 1 | 2 | [Contrastive ultrasound](batch-80-results.md) |
| 100 | 20 | 40 | [Larger models and method comparisons](batch-100-results.md) |

The first 30 professor pairs appear in the portfolio repository’s earlier project folders and standalone repositories. The sources in this directory preserve the settings used to generate the published metrics.

## Re-running a study

Follow the command in the study's README and install the versions listed in its requirements file. The simulated studies generate their inputs. Public-data studies expect local copies of the referenced datasets. See the [shared data layout](batch100/README.md) for ICU, EEG, diabetes, activity, retinal, CT, and radiology data. Additional inputs are listed below.

- `medmnist/pneumoniamnist.npz`, `medmnist/pneumoniamnist_64.npz`, `medmnist/pathmnist.npz`, and `medmnist/breastmnist.npz` come from the [MedMNIST+ release](https://zenodo.org/records/10519652).
- `sleep-accel/heart_rate/` and `sleep-accel/labels/` contain [PhysioNet wearable sleep recordings](https://physionet.org/content/sleep-accel/1.0.0/).
- `lung-survival.csv` is the [NCCTG lung cancer dataset](https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/datasets/lung.csv).
- `mini-speech-commands.zip` is the [TensorFlow Mini Speech Commands archive](https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip).
- `malaria-cell-images.zip` is the [NIH malaria crop archive](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip).
- The caption-negation study reads `real-data-projects/roco-radiology-retrieval/data/sample_manifest.csv`, which the [pinned radiology feature preparation script](../crossmodal-pair-audit-real-data/prepare.py) generates alongside the frozen embeddings. Use the same sampled captions and official train/test labels.

Training writes weights, split information, and updated metrics locally. The published results are tables and aggregate metrics.
