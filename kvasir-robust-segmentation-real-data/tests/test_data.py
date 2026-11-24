from pathlib import Path
import cv2
import numpy as np
import pytest
from kvasir_robust.data import discover_pairs, fold_indices


def test_official_dataset_is_complete():
    root=Path("D:/USA based documents/Haider Heavy Research Projects/datasets/kvasir-seg/extracted/Kvasir-SEG")
    assert len(discover_pairs(root))==1000


def test_folds_do_not_overlap():
    train,val,test=fold_indices(1000,5,27,0)
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    assert len(train)+len(val)+len(test)==1000


def test_empty_masks_are_rejected(tmp_path):
    (tmp_path/"images").mkdir(); (tmp_path/"masks").mkdir()
    cv2.imwrite(str(tmp_path/"images"/"sample.png"), np.zeros((8,8,3), np.uint8))
    cv2.imwrite(str(tmp_path/"masks"/"sample.png"), np.zeros((8,8), np.uint8))
    with pytest.raises(ValueError, match="empty masks"):
        discover_pairs(tmp_path)

