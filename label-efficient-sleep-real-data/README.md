# Sleepâ€“wake classification with sparse labels (public data)

AUROC (labeled-only vs pseudo-label training): 0.7062 vs 0.7091, averaged over three seeds.

![Mean held-out scores](comparison.svg)

## Study design

Hold out one quarter of participants. Train a 96-tree forest with 10% of the training epochs labeled. Retrain after adding training-only pseudo-labels with probability below 0.08 or above 0.92. Metrics pool held-out epochs within each split. PhysioNet provides Apple Watch heart-rate samples and polysomnography stage labels. Aggregate heart-rate level and variability over two windows ending at the end of each 30-second epoch. Drop epochs without enough aligned readings. The target is sleep versus wake. The experiment uses heart rate alone.

Independent portfolio experiment inspired by work from Prashant Shenoy. [Research source](https://lass.cs.umass.edu/projects/mobile-health.html)

[Public dataset](https://physionet.org/content/sleep-accel/1.0.0/)

## Results

| Method | auroc | average_precision | balanced_accuracy | test_subjects |
|---|---:|---:|---:|---:|
| labeled only | 0.7062 | 0.9506 | 0.6178 | 8.0000 |
| pseudo labels | 0.7091 | 0.9511 | 0.6227 | 8.0000 |

Per-seed results are in [results.json](results.json).

## Reproduce

Local CPU run, Python 3.10. Model checkpoints are generated during training. Set PORTFOLIO_DATA to reuse an existing data directory.

```sh
python -m pip install -r requirements.txt
python prepare.py
python train.py
```

[Training notebook](reproduce.ipynb)
