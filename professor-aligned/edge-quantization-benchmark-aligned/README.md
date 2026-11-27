# Yi Sheng — Professor-aligned simulation

Train a two-hidden-layer MLP with early stopping. Compare its float32 weights with 8-bit and 4-bit per-channel weight quantization. Evaluate using dequantized CPU arithmetic. Payload figures model packed weights plus float32 scales and biases and exclude the scaler and runtime. No device latency was measured.

This experiment uses generated data with known labels or trajectories.

[Research reference](https://eaho-research-lab.github.io/Homepage/)

## Mean held-out results

| Arm | auroc | average_precision | parameter_payload_bytes | disagreement |
|---|---:|---:|---:|---:|
| fp32 | 0.97445 | 0.97868 | 31044 | 0 |
| int8_weights | 0.97437 | 0.9786 | 8216 | 0.00066667 |
| int4_weights | 0.97445 | 0.97874 | 4368 | 0.01 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
