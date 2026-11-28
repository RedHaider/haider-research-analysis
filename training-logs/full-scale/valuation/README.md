# Out-of-bag data valuation

James Zou co-authored [Data-OOB](https://proceedings.mlr.press/v202/kwon23e.html), which scores examples using the base learners that did not train on those examples. This independent benchmark compares OOB vote agreement, a within-class rank of OOB probabilities, and random ranking. The proposed rank is designed to reduce the effect of systematically lower model confidence for some observed classes.

The aligned study uses 36,000 simulated records. The public-data benchmark uses all 581,012 records of [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype), grouping identical feature profiles across partitions. Training has seeded, feature-dependent label flips. Validation and test labels come from the original source. The benchmark reports detection AP and precision at the fixed review budget and retrains gradient-boosted trees to see how each pruning strategy affects held-out performance. Flip detection refers to the injected errors.

    python training-logs/full-scale/valuation/train.py --threads 1
    python training-logs/full-scale/valuation/train.py --real --threads 2

The trainer reads datasets/covertype.zip for the public-data experiment. Install NumPy, pandas, scikit-learn, joblib, and threadpoolctl. Split indices, scores, labeled flip indicators, per-arm trained models, and metrics are saved under the respective project directory.
