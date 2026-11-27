# Ju Sun — Professor-aligned simulation

Train a pooled logistic model and a site-and-class reweighted variant. Hold out each site in turn, use only source-site data for fitting, and report mean and worst-site discrimination plus mean Brier score. This tests weighting under distribution shift.

This experiment uses generated data with known labels or trajectories.

[Research reference](https://cse.umn.edu/cs/news/ju-sun-part-12m-grant-study-role-ai-breast-cancer-treatment)

## Mean held-out results

| Arm | auroc | average_precision | brier | worst_site_auroc |
|---|---:|---:|---:|---:|
| pooled | 0.72725 | 0.55588 | 0.19621 | 0.66543 |
| site_class_weighted | 0.73989 | 0.56579 | 0.21744 | 0.69609 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
