from probabilistic_seg.experiment import sample,evaluate
def test_shape():
 t,a=sample(2,5,32);assert t.shape==(5,32,32) and a.shape==(5,5,32,32)
def test_metrics():assert all(v>=0 for v in evaluate(3).values())

