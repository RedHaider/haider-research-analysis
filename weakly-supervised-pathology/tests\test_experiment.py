from pathology_mil.experiment import bags,evaluate
def test_shape():
 x,y=bags(20,2);assert x.shape==(20,30,6) and len(y)==20
def test_metrics():assert all(0<=v["auroc"]<=1 for v in evaluate(3).values())

