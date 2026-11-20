from selective_shift.experiment import evaluate
def test_metrics():
 r=evaluate(3);assert 0<=r['auroc']<=1 and all(0<=v['coverage']<=1 for k,v in r.items() if k!='auroc')
