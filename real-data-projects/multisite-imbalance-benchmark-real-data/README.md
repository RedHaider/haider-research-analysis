# Ju Sun — Public-data companion

Train a pooled logistic model and a site-and-class reweighted variant. Hold out each site in turn, use only source-site data for fitting, and report mean and worst-site discrimination plus mean Brier score. This tests weighting under distribution shift.

UCI Heart Disease, four hospitals and 920 records. The outcome is recorded heart disease. This public-data analogue tests site and class weighting for a different endpoint from cardiotoxicity.

[Research reference](https://cse.umn.edu/cs/news/ju-sun-part-12m-grant-study-role-ai-breast-cancer-treatment)

## Mean held-out results

| Arm | auroc | average_precision | brier | worst_site_auroc |
|---|---:|---:|---:|---:|
| pooled | 0.78864 | 0.88714 | 0.17114 | 0.675 |
| site_class_weighted | 0.72894 | 0.82627 | 0.20244 | 0.63852 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
