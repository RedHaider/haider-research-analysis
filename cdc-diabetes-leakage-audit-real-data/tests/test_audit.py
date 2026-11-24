import numpy as np
from real_leakage.audit import moment_match


def test_moment_matching_equalizes_group_means():
    rng=np.random.default_rng(1); group=np.repeat([0,1],100)
    data=rng.normal(size=(200,4))+group[:,None]*2
    transformed,_=moment_match(data,data,group,group)
    assert np.max(np.abs(transformed[group==0].mean(0)-transformed[group==1].mean(0)))<1e-6


def test_shape_is_preserved():
    data=np.arange(24,dtype=float).reshape(6,4); group=np.array([0,0,0,1,1,1])
    train,test=moment_match(data,data,group,group)
    assert train.shape==data.shape and test.shape==data.shape

