from ultrasound_multi.experiment import data,evaluate
def test_shape():assert data(20,2)[0].shape==(20,10)
def test_metrics():
 r=evaluate(3);assert 0<=r['malignancy_auroc']<=1 and 0<=r['category_accuracy']<=1

