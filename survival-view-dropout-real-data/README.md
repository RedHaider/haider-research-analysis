# Missing-view survival prediction (public data)

Harrell concordance (ordinary training vs view dropout with one view withheld): 0.623 vs 0.619, averaged over three seeds.

![Mean held-out scores](comparison.svg)

## Study design

Fit discrete-time logistic hazards across six intervals, with and without duplicating training records after masking the second feature group. Test on complete and masked inputs. Score one-year predicted risk using Harrell concordance across all comparable follow-up pairs. The NCCTG lung cancer cohort contains 228 records with censoring. Each split holds out 30% of records. The missing view comprises patient-reported performance, calories and weight loss. Both views are tabular clinical variables.

Independent portfolio experiment inspired by work from Junzhou Huang. [Research source](https://ranger.uta.edu/~huang/Publication.htm)

[Public dataset](https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/datasets/lung.csv)

## Results

| Method | concordance | comparable_pairs |
|---|---:|---:|
| complete training available | 0.6417 | 1792.6667 |
| complete training missing | 0.6234 | 1792.6667 |
| view dropout available | 0.6310 | 1792.6667 |
| view dropout missing | 0.6188 | 1792.6667 |

Per-seed results are in [results.json](results.json).

## Reproduce

Local CPU run, Python 3.10. Model checkpoints are generated during training. Set PORTFOLIO_DATA to reuse an existing data directory.

```sh
python -m pip install -r requirements.txt
python prepare.py
python train.py
```

[Training notebook](reproduce.ipynb)
