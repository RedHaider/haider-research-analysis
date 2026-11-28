# Fractional-memory signal forecasting

[Quantification of Fractional Dynamical Stability of EEG Signals](https://www.frontiersin.org/journals/control-engineering/articles/10.3389/fcteg.2021.787747/full), co-authored by Paul Bogdan, motivates modeling long memory in physiological sequences. This independent forecast benchmark tests a bank of causal truncated Grünwald–Letnikov transforms added to a residual MLP and a ridge model. Short- and long-history baselines use raw taps. The experimental endpoint is multi-horizon forecast error, distinct from the original paper's stability certificate.

The aligned study uses motor-imagery run R04 from [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/) and nine electrodes common to the recordings. It excludes sessions with a nonstandard sample rate. The companion uses the [Fordham WISDM Activity Prediction dataset](https://www.cis.fordham.edu/wisdm/dataset.php). It separates people across partitions and respects timestamp discontinuities and activity boundaries when creating windows. All normalizers fit on training subjects, and only causal measurements are inputs. Checkpoints and raw-unit forecast errors for four horizons are retained.

    python training-logs/full-scale/fractional/fetch_eeg.py
    python training-logs/full-scale/fractional/train.py
    python training-logs/full-scale/fractional/train.py --real

Install NumPy, pandas, SciPy, scikit-learn, PyTorch, joblib and threadpoolctl. Real-data companion expects datasets/wisdm-activity-v1.1.tar.gz. The dataset progress output verifies the published checksum for all downloaded EEG files.
