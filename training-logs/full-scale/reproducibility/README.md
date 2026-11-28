# Reproducibility of paired high-throughput measurements

[Measuring Reproducibility of High-Throughput Experiments](https://arxiv.org/abs/1110.4705), co-authored by Qunhua Li, motivates classifying paired scores through a rank-based mixture. This independent approximation fits two Gaussian components after training-frozen marginal rank normalization. It compares a fixed background with an empirical background estimated from lower scores. Both are compared with minimum and geometric-mean rank baselines. It does not implement the original iterative copula margins.

The aligned experiment simulates concordant effects, chromosome shifts, and one-replicate outliers. The real study uses two UW replicates of [K562 CTCF ChIP-seq](https://www.encodeproject.org/experiments/ENCSR000DWE/) and independent [Stanford peak calls](https://www.encodeproject.org/experiments/ENCSR000EGM/) deposited at UCSC. Candidate bins come only from the two UW tracks. A Stanford summit within 350 bases of a bin center supplies the evaluation reference. Four folds hold out whole chromosomes. Concordance with an independent peak call is the test endpoint. Local posterior values are not calibrated false discovery rates.

    python training-logs/full-scale/reproducibility/train.py
    python training-logs/full-scale/reproducibility/train.py --real

Install NumPy, pandas, SciPy, scikit-learn, and threadpoolctl. The real run expects the two UW narrowPeak files and one Stanford narrowPeak file under datasets/encode-ctcf. Trained parameters, frozen reference margins and held-out chromosome-level rankings are stored with the reports.
