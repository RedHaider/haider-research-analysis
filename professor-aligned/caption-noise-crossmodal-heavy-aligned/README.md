# Imageâ€“text matching with degraded captions — Professor-aligned simulation

Verify radiology image–caption pairs using existing frozen embeddings. Zero a quarter of caption embedding coordinates in the stress condition. The larger verifier trains with a mix of unaltered and altered embeddings.

For this run the source observations and labels are simulated.
Input shape is 2048. There are 3,520 observations before splitting.

[Research connection](https://engineering.uark.edu/electrical-engineering-computer-science/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.98672 | 0.99951 | 0.99859 | 0.0084671 | 1.3347e+05 | 2.2461 | 3 | 7 |
| compact_stress | 0.98438 | 0.99737 | 0.99404 | 0.01358 | — | — | — | — |
| fixed_reference | 0.27344 | 0.99993 | 0.99981 | 0.24118 | — | — | — | — |
| heavy | 0.98828 | 0.99909 | 0.9974 | 0.010138 | 2.2346e+06 | 13.058 | 2.5 | 6.5 |
| heavy_stress | 0.97813 | 0.99514 | 0.98927 | 0.020866 | — | — | — | — |

Run `python ../../training-logs/batch100/train.py 16` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
