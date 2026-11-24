import numpy as np
from kvasir_prob.features import pixel_features
def test_shape_and_range():
    f=pixel_features(np.zeros((20,30,3),dtype=np.uint8),32);assert f.shape==(1024,8);assert np.all((f>=0)&(f<=1))

