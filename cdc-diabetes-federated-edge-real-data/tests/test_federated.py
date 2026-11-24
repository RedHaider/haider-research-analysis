import numpy as np

from federated_edge.federated import dequantize_int8, quantize_int8, sigmoid


def test_int8_round_trip_has_small_error():
    update = np.linspace(-1, 1, 101)
    quantized, scale = quantize_int8(update)
    reconstructed = dequantize_int8(quantized, scale)
    assert np.max(np.abs(update - reconstructed)) <= scale / 2 + 1e-7


def test_zero_update_is_supported():
    quantized, scale = quantize_int8(np.zeros(8))
    assert np.all(dequantize_int8(quantized, scale) == 0)


def test_sigmoid_is_bounded():
    result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all((result >= 0) & (result <= 1))
    assert result[1] == 0.5

