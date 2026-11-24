from uci_heart_robust.metrics import binary_metrics, expected_calibration_error


def test_perfect_predictions():
    metrics = binary_metrics([0.01, 0.05, 0.95, 0.99], [0, 0, 1, 1])
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["brier"] < 0.01


def test_ece_includes_endpoints():
    assert expected_calibration_error([0.0, 1.0], [0, 1]) == 0.0


def test_all_metrics_are_bounded():
    assert all(0 <= value <= 1 for value in binary_metrics([0.2, 0.8, 0.4, 0.7], [0, 1, 1, 0]).values())

