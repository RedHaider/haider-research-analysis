# Landmark graphs for sparse subspace clustering

[Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit](https://openaccess.thecvf.com/content_cvpr_2016/html/You_Scalable_Sparse_Subspace_CVPR_2016_paper.html) by You, Robinson, and Vidal motivates representing each sample with a few related atoms. This independent inductive version restricts OMP to similar atoms selected from a fixed training dictionary and uses a sparse bipartite graph so new observations can be embedded without refitting. Baselines are PCA with mini-batch k-means and a cosine-neighbor version of the same graph pipeline.

The aligned study uses a simulated union of eight partly overlapping subspaces. The public-data study uses the official 60,000/10,000 split of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). Model fitting is unsupervised. Labels are used only to evaluate clusters and establish the training-to-label mapping for the accuracy metric. Evaluation reports ARI, NMI, and mapped accuracy on clean and perturbed held-out records. Perturbations are artificial horizontal occlusion and noise for Fashion-MNIST and additive noise for the simulation.

    python training-logs/full-scale/subspaces/train.py --threads 1
    python training-logs/full-scale/subspaces/train.py --real --threads 1

Install NumPy, SciPy, scikit-learn, joblib, and threadpoolctl. The real run reads the four compressed IDX files under datasets/fashion-mnist. Fitted PCA, dictionary and spectral projection, cluster assignments via the saved estimators, and results stay on D:.
