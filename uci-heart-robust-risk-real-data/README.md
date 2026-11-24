# UCI Heart Disease Robust Risk Benchmark

Real-data companion to the professor-aligned cardiovascular-risk project for
Juhwan Lee. It uses the public UCI Cleveland Heart Disease dataset and compares
a conventionally trained logistic model with a measurement-error-aware model
trained on noisy and partially missing replicas.

The benchmark uses repeated stratified train/validation/test splits, held-out
Platt calibration, and clean, noise, missing-value, and combined-shift tests.
It reports AUROC, AUPRC, Brier score, balanced accuracy and calibration error.

Run `run_benchmark.py` or open `UCI_Heart_Robust_Risk_Colab.ipynb`. The raw CSV,
results and all caches are written under drive D. This is a portfolio benchmark,
not a clinical system or a reproduction of Professor Lee's published model.

## Validated results

Across 20 repeated held-out splits, the proposed five-member robust ensemble
increased mean shifted AUROC from **0.8754 to 0.8837** and reduced mean shifted
Brier score from **0.1457 to 0.1396**. The paired AUROC gain was +0.0083
(95% CI 0.0042 to 0.0124; positive in 15/20 splits), while the paired Brier
reduction was 0.0061 (95% CI 0.0042 to 0.0081; positive in 19/20 splits).
Clean AUROC remained similar at 0.8984 versus 0.8998.

