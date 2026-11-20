from limited_ct.experiment import phantom,evaluate
def test_shape():assert phantom(2,32).shape==(32,32)
def test_metrics():
 r=evaluate(3);assert all(v["rmse"]>=0 and -1<=v["correlation"]<=1 for v in r.values())
