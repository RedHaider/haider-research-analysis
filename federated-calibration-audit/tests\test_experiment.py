from fed_calibration.experiment import clients,evaluate
def test_clients():assert len(clients(2,50))==5
def test_metrics():assert all(v>=0 for v in evaluate(3).values())

