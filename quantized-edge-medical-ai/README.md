# Quantized Edge Medical AI

Synthetic resource-constrained medical classification benchmark. It quantizes
a trained linear model to 16, 8, and 4 bits and measures AUROC, model size, and
prediction disagreement relative to full precision.

Run: `python -m edge_quant.experiment --output outputs/demo`
