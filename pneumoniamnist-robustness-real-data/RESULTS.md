# Validated three-seed results

| Condition | Baseline AUROC | Proposed AUROC | Absolute change |
| --- | ---: | ---: | ---: |
| Clean | 0.9412 | 0.9423 | +0.0010 |
| Gaussian noise | 0.6666 | 0.9260 | +0.2594 |
| Blur | 0.8647 | 0.9440 | +0.0793 |
| Contrast | 0.8228 | 0.9357 | +0.1129 |
| Brightness | 0.8493 | 0.8871 | +0.0378 |
| Four-corruption mean | 0.8008 | 0.9232 | **+0.1224** |

The proposed method also reduced clean ECE from 0.1261 to 0.0650 and increased
clean balanced accuracy from 0.7625 to 0.8312. Clean F1 decreased from 0.8746
to 0.8173. Therefore the defensible claim is improved corruption robustness,
calibration and class-balanced performance—not superiority on every metric.

Protocol: PneumoniaMNIST 28-pixel official splits, three seeds (17, 29, 43),
matched residual CNNs, six-epoch maximum, early stopping, balanced sampling,
and four prespecified corruptions. This is not a clinical claim or a reproduction
of Professor Truong-Son Hy's published system.

