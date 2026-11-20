from glaucoma_proto.experiment import data,evaluate
def test_shape():
 x,y=data(20,2);assert x.shape==(20,12) and len(y)==20
def test_metrics():assert all(0<=v<=1 for v in evaluate(3).values())

