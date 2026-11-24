# Validated results

| Evaluation | Baseline | Proposed | Change |
| --- | ---: | ---: | ---: |
| Clean AUROC | 0.8984 | 0.8998 | +0.0014 |
| Clean Brier | 0.1304 | 0.1277 | -0.0027 |
| Shifted mean AUROC | 0.8754 | 0.8837 | **+0.0083** |
| Shifted mean Brier | 0.1457 | 0.1396 | **-0.0061** |

The paired shifted-AUROC improvement had a 95% confidence interval of 0.0042
to 0.0124 and was positive in 15 of 20 splits. The paired shifted-Brier reduction
had a 95% confidence interval of 0.0042 to 0.0081 and was positive in 19 of 20
splits.

Protocol: 20 repeated stratified 60/20/20 splits of the 303-participant UCI
Cleveland Heart Disease dataset; 13 features; held-out Platt calibration;
measurement noise, missingness and combined shifts; five independently augmented
members in the proposed ensemble. Shifts are simulated, and this is not a clinical
claim or a reproduction of Professor Juhwan Lee's model.

