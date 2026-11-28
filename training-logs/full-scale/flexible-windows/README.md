# Flexible observation and forecast windows

[Flexible-Window Predictions on Electronic Health Records](https://pubmed.ncbi.nlm.nih.gov/36312212/), co-authored by Mehak Gupta, motivates one model capable of answering different observation/forecast window requests. This independent benchmark compares window-specific boosted trees, a shared two-layer GRU with independent outputs, and a GRU that accumulates projected hourly risk to guarantee monotonicity across longer horizons.

The real-data project uses the 12,000 [PhysioNet 2012 ICU records](https://physionet.org/content/challenge-2012/1.0.0/). It fits and validates with separate patients from set A, then evaluates on B and C. It predicts whether averaged invasive or non-invasive MAP measurements go below 65 during selected future windows. It includes only target windows with adequate future BP measurement coverage. Missingness and elapsed time are explicit inputs. The simulated companion tests the same models with randomized missing observations under longitudinal drift. This benchmark measures forecast discrimination, calibration and consistency across the windows.

    python training-logs/full-scale/flexible-windows/train.py
    python training-logs/full-scale/flexible-windows/train.py --real

Install NumPy, PyTorch, scikit-learn, joblib and threadpoolctl. The real-data archive files live under datasets/icu2012. Split IDs, parameters, held-out predictions, eligibility masks and per-window reports remain on D:.
