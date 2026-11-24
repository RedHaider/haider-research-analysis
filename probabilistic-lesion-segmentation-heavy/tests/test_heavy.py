from probabilistic_seg.heavy_experiment import dice,reliability_weights,simulate
import numpy as np
def test_dice_identity():
    x=np.eye(8,dtype=bool);assert dice(x,x)==1.0
def test_shapes():
    truth,annotations=simulate(1,n=3,size=32);assert truth.shape==(3,32,32) and annotations.shape==(3,7,32,32)
def test_weights_sum_to_one():
    _,annotations=simulate(2,n=10,size=32);assert abs(reliability_weights(annotations).sum()-1)<1e-9

