from wearable_readmit.experiment import data,evaluate
def test_shape():
 x,y=data(20,2);assert x.shape==(20,12,4) and len(y)==20
def test_metrics():assert all(0<=v['auroc']<=1 for v in evaluate(3).values())

