# Weak-label pathology bag classification

Saeed Amal's [research profile](https://coe.northeastern.edu/people/amal-saeed/) describes image-based cancer research including breast tumor histopathology. This independent project tests mean pooling and [gated attention](https://arxiv.org/abs/1802.04712) with a shared residual tile encoder trained end to end under bag-level supervision.

* The aligned benchmark uses generated tissue tiles.
* The companion uses [PathMNIST](https://medmnist.com/), whose real colorectal histology patches are combined into constructed 12-tile bags. All bags stay inside their original dataset split, and a tile appears in at most one bag per split. Positive bags contain malignant epithelium. Constructed bags are a controlled weak-label benchmark. Source slide identifiers are unavailable in the archive, and the project does not make whole-slide or patient-level performance claims.

    python training-logs/full-scale/pathology/train.py --epochs 10 --threads 3
    python training-logs/full-scale/pathology/train.py --real --epochs 10 --threads 3

Install PyTorch, NumPy, and scikit-learn. The real benchmark reads datasets/medmnist/pathmnist.npz. Each arm is trained with two seeds. Validation average precision determines checkpoint selection. The final report records independent held-out AUROC and average precision for every arm and seed. Checkpoints and bag indices are retained on D:.
