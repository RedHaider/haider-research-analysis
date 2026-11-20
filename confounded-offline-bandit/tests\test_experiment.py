from offline_bandit.experiment import data,evaluate
def test_shape():
 x,a,r,t,b=data(20,2);assert x.shape==(20,6) and all(len(v)==20 for v in [a,r,t,b])
def test_metrics():assert all(v["regret"]>=0 and 0<=v["oracle_agreement"]<=1 for v in evaluate(3).values())

