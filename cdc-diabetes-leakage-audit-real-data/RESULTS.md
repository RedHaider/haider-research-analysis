# Validated real-data leakage audit

| Representation | Disease AUROC | Linear sex-probe AUROC | Nonlinear sex-probe AUROC |
| --- | ---: | ---: | ---: |
| Original | 0.8113 | 0.5762 | 0.6109 |
| Linear projection | 0.8113 | 0.5723 | 0.6065 |
| Moment matching | 0.8110 | 0.5030 | **1.0000** |
| Quantile matching | 0.8057 | 0.5557 | 0.6901 |

The audit demonstrates why a linear probe alone is insufficient. Moment matching
appeared successful under the linear probe while introducing a representation
that a nonlinear probe separated perfectly. No mitigation produced a large,
credible multi-probe reduction; the negative result is retained rather than
overstated.

