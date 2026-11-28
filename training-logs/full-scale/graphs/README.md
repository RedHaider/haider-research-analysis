# Label-conditioned graph message passing

[Label-Wise Graph Convolutional Network for Heterophilic Graphs](https://proceedings.mlr.press/v198/dai22b.html), co-authored by Suhang Wang, motivates preserving different neighbor contexts. This implementation compares a feature-only residual network, mean graph aggregation, hard neighbor class groups, and posterior-weighted neighbor groups. Training-node class posteriors are out of fold. Validation and test posteriors come from the training classifiers. The same residual readout architecture processes each feature bank. The experiment uses transductive connectivity with only training labels available to the fitted models.

The aligned study builds a 24,000-node graph with asymmetric cross-class edge frequencies. The real-data companion uses the [Amazon Ratings graph published by Platonov et al.](https://github.com/yandex-research/heterophilous-graphs), which contains 24,492 nodes and 93,050 undirected edges. It evaluates the first three official masks. Metrics include macro-F1 and log loss to capture class balance and probabilistic quality.

    python training-logs/full-scale/graphs/train.py
    python training-logs/full-scale/graphs/train.py --real

The companion reads datasets/amazon_ratings.npz. Install NumPy, SciPy, scikit-learn, PyTorch and threadpoolctl. The output stores posteriors' regression parameters, PCA, masks, checkpoint weights, predictions and held-out metrics.
