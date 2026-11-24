from __future__ import annotations

import numpy as np


def sigmoid(value):
    value = np.clip(value, -30, 30)
    return 1.0 / (1.0 + np.exp(-value))


def quantize_int8(update):
    update = np.asarray(update, dtype=float)
    scale = max(float(np.max(np.abs(update))) / 127.0, 1e-12)
    return np.clip(np.round(update / scale), -127, 127).astype(np.int8), scale


def dequantize_int8(quantized, scale):
    return np.asarray(quantized, dtype=np.float32) * float(scale)

