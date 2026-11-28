# Frozen nuisance correction for expression prediction

[Removing batch effects for prediction problems with frozen surrogate variable analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4179553/) motivates training-only estimation of nuisance correction for future observations. This independent experiment learns a low-rank nuisance projection from outcome-residualized training features. It compares raw logistic regression, a raw residual MLP, the same MLP after frozen projection, and a control-feature correction arm. The simulation uses known controls. METABRIC uses training-derived low-association features as proxies. This simplified estimator differs from the original weighted fSVA algorithm.

The aligned benchmark uses five simulated cohorts with a sign reversal in a nuisance–outcome association. The public-data companion uses [METABRIC from cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric) and predicts ER status, holding out one entire study cohort at a time. Cohort variation can include both biological and technical differences. The input removes the ESR1 probe as a direct marker check. Transforms and feature selection are refit from scratch on the training patients of each rotation. Each validation partition selects a checkpoint, and each test cohort yields AUROC, average precision, and Brier score.

    python training-logs/full-scale/surrogate/train.py --threads 1
    python training-logs/full-scale/surrogate/train.py --real --threads 1

The second command reads the expression and metadata files from the processed [METABRIC deposit](https://zenodo.org/records/18272235). Install NumPy, pandas, PyTorch, and scikit-learn. Each rotation stores split IDs, preprocessing and nuisance projection arrays, model weights, predictions, labels, and test metrics. The results JSON summarizes the rotation-level performance.
