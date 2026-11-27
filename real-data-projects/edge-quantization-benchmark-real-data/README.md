# Yi Sheng — Public-data companion

Train a two-hidden-layer MLP with early stopping. Compare its float32 weights with 8-bit and 4-bit per-channel weight quantization. Evaluate using dequantized CPU arithmetic. Payload figures model packed weights plus float32 scales and biases and exclude the scaler and runtime. No device latency was measured.

Official PneumoniaMNIST training and test partitions, 4,708 and 624 images. Downsample to 12 × 12 before training.

[Research reference](https://eaho-research-lab.github.io/Homepage/)

## Mean held-out results

| Arm | auroc | average_precision | parameter_payload_bytes | disagreement |
|---|---:|---:|---:|---:|
| fp32 | 0.92932 | 0.92257 | 31044 | 0 |
| int8_weights | 0.92934 | 0.9227 | 8216 | 0.0016026 |
| int4_weights | 0.92976 | 0.92367 | 4368 | 0.0069444 |

Run with the same Python environment and set `PORTFOLIO_DATA` to the dataset directory before running `python train.py`. Per-seed results are in `results.json` and trained models are in `checkpoints/`.
