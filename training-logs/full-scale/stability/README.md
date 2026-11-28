# Clustering validation for expression data

[clValid](https://www.jstatsoft.org/article/view/v025i04), co-authored by Guy Brock, motivates comparing internal and stability scores for choosing an unsupervised model. This independent project compares validation silhouette with the lower quartile of adjusted Rand agreement under cell resampling and gene subsampling. It benchmarks two clustering families and five cluster counts in an SVD feature space fit on training data.

The aligned study uses generated count data with known cell groups, unequal frequencies, library-size variation, and batch effects. Its real-data companion uses [10x Genomics Fresh 68k PBMCs, Donor A](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0). Both evaluate selection on held-out cells with a gene-dropout stress test. The real dataset contains cells from one donor, so the holdout estimates within-donor agreement. The scores do not establish biological cell types.

    python training-logs/full-scale/stability/train.py
    python training-logs/full-scale/stability/train.py --real

Install NumPy, SciPy, scikit-learn, joblib and threadpoolctl. The companion reads datasets/pbmc68k-filtered.tar.gz. It saves split indices, model-selection scores, selected models and held-out cluster assignments.
