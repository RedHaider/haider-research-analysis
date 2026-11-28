# Extreme-tail scoring for open-set image recognition

[The Extreme Value Machine](https://pubmed.ncbi.nlm.nih.gov/28541894/), co-authored by Walter Scheirer, motivates using extreme cross-class margins to recognize inputs from unseen classes. This independent experiment trains a residual encoder on known classes, fits Weibull distributions to nearby nonmatch distances from sampled anchors and compares survival-score ranking with maximum softmax on the same encoder. The anchors use a fixed random budget per known class. The original EVM's cover-set selection is outside this implementation.

The aligned benchmark uses generated texture families. The real benchmark uses [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), with three known fine classes per superclass and distinct classes for calibration diagnostics and final unknown testing. Only known-class training data fit the encoder and Weibull tails. The rejection threshold is the 5th percentile of known validation confidence. The report includes closed-set accuracy, unknown AUROC, test false acceptance at the frozen threshold, and open-set classification rate.

    python training-logs/full-scale/open-set/train.py --threads 1
    python training-logs/full-scale/open-set/train.py --real --threads 2

Install NumPy, SciPy, scikit-learn, PyTorch and threadpoolctl. The public-data run also requires Pillow and PyArrow. It reads the train and test Parquet files from the [University of Toronto CIFAR-100 mirror](https://huggingface.co/datasets/uoft-cs/cifar100) saved as datasets/cifar100-train.parquet and datasets/cifar100-test.parquet. Checkpoints include the optimizer and random-number state to resume on D:. The outputs include the exact split, Weibull tails, and held-out scores.
