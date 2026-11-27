# Zijun Yao — Professor-aligned simulation

Predict normalized brain volume at the next observed visit from at least two prior observations. Inject a reproducible measurement perturbation into one prior observation in 40% of trajectories. Compare a latest-value and OLS summary with a Theil–Sen trend summary with fixed shrinkage toward zero for short observation windows using the same ridge model and person-level out-of-fold predictions.

This experiment uses generated data with known labels or trajectories.

[Research reference](https://ittc.ku.edu/~zyao/publications/)

## Mean held-out results

| Arm | mae | subjects |
|---|---:|---:|
| latest_ols | 0.014109 | 600 |
| robust_trend | 0.013152 | 600 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
