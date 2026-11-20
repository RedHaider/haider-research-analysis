from experiment import run

def test_metrics_are_valid():
    report = run(3)
    assert all(0 <= values["auroc"] <= 1 for values in report.values())

