from pathlib import Path
from kvasir_robust.data import discover_pairs, fold_indices


def test_official_dataset_is_complete():
    root=Path("D:/USA based documents/Haider Heavy Research Projects/datasets/kvasir-seg/extracted/Kvasir-SEG")
    assert len(discover_pairs(root))==1000


def test_folds_do_not_overlap():
    train,val,test=fold_indices(1000,5,27,0)
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    assert len(train)+len(val)+len(test)==1000
