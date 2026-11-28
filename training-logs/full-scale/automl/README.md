# Frugal AutoML with temporal validation

[FLAML](https://proceedings.mlsys.org/paper_files/paper/2021/hash/1ccc3bfa05cb37b917068778f3c4523a-Abstract.html), co-authored by Qingyun Wu, motivates using cheaper trials to allocate resources. This independent project compares shuffled full-data search against a three-stage fidelity schedule over Extra Trees and histogram gradient boosting. Both arms share one nominal budget and common candidate order. Fit cost units approximate sample count, training iterations, and capacity. The trial logs preserve measured CPU time to test how well those units reflect work.

The aligned experiment uses a simulated chronological drift benchmark. The companion uses the 45,312-record [OpenML Electricity dataset](https://www.openml.org/d/151). The source contains out-of-order records and duplicate timestamps. The trainer drops all records sharing ambiguous timestamps, sorts the rest by the supplied day and period, and excludes contemporaneous price and demand from predictors. Train and validation are cut by time with a gap before subsequent blocks. It uses lagged market series and calendar features. In this retrospective setting lag availability and benchmark-wide normalization follow the provided data. All splits, candidate trials, refit models, predictions, and held-out metrics are saved.

    python training-logs/full-scale/automl/train.py --budget-multiple 18
    python training-logs/full-scale/automl/train.py --real --budget-multiple 18

Install NumPy, pandas, SciPy, scikit-learn, joblib and threadpoolctl. The companion expects datasets/electricity-151.arff.
