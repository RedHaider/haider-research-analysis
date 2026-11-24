import numpy as np

from pneumoniamnist_robust.metrics import binary_metrics, expected_calibration_error


def test_perfect_predictions_have_perfect_discrimination():
    metrics = binary_metrics([0.01, 0.05, 0.95, 0.99], [0, 0, 1, 1])
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["f1"] == 1.0


def test_ece_includes_probability_one():
    assert expected_calibration_error(np.array([0.0, 1.0]), np.array([0, 1])) == 0.0


def test_metric_ranges():
    metrics = binary_metrics([0.2, 0.7, 0.4, 0.8], [0, 1, 1, 0])
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

