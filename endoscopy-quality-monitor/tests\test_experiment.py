from endoscopy_quality.experiment import evaluate
def test_metrics():assert all(0<=v<=1 for v in evaluate(3).values())

