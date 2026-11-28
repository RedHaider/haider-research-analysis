import numpy as np
import pandas as pd
from common import DATA, save
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

MEASURES = ("P_Area", "Q_Area", "R_Area", "S_Area", "T_Area", "QRS_Area")
SEEN = ("I", "II", "V1", "V2")
MISSING = ("V3", "V4", "V5", "V6")


def inputs(real):
    if real:
        features = [f"{kind}_{lead}" for lead in (*SEEN, *MISSING) for kind in MEASURES]
        records = pd.read_csv(
            DATA / "ptbxl-records.csv", usecols=["ecg_id", "patient_id", "strat_fold"]
        )
        signal = pd.read_csv(
            DATA / "ptbxl-features.csv", usecols=["ecg_id", *features]
        ).merge(records, on="ecg_id", validate="one_to_one")
        signal = signal.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
        x = signal[features].to_numpy(dtype=np.float32)
        train = np.flatnonzero(signal.strat_fold.to_numpy() <= 8)
        test = np.flatnonzero(signal.strat_fold.to_numpy() == 10)
        assert not set(signal.patient_id.iloc[train]) & set(
            signal.patient_id.iloc[test]
        )
        return x, train, test, signal.ecg_id.to_numpy()
    rng = np.random.default_rng(95)
    latent = rng.normal(0, 1, (24000, 10))
    mapping = rng.normal(0, 1, (10, 48))
    torso = rng.uniform(0.65, 1.5, (len(latent), 1))
    x = latent @ mapping + np.sin(latent @ mapping * 0.4) * torso
    x += rng.normal(0, 0.45, x.shape)
    return (
        x.astype(np.float32),
        np.arange(18000),
        np.arange(18000, 24000),
        np.arange(24000),
    )


def run(real):
    x, train, test, ids = inputs(real)
    split_point = len(SEEN) * len(MEASURES)
    visible, target = x[:, :split_point], x[:, split_point:]
    p = RobustScaler().fit(visible[train])
    q = RobustScaler().fit(target[train])
    u, v = p.transform(visible).clip(-10, 10), q.transform(target).clip(-12, 12)
    report, models, estimates = {}, {}, {}
    for name, model in (
        ("ridge", Ridge(alpha=10)),
        (
            "nonlinear_latent",
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 96),
                max_iter=50,
                early_stopping=True,
                batch_size=256,
                random_state=95,
            ),
        ),
    ):
        model.fit(u[train], v[train])
        estimate = model.predict(u[test])
        delta = (estimate - v[test]).reshape(-1, len(MISSING), len(MEASURES))
        report[name] = {
            "robust_scaled_rmse": float(np.sqrt(np.mean(delta**2))),
            "per_lead_rmse": dict(
                zip(MISSING, np.sqrt(np.mean(delta**2, axis=(0, 2))).tolist())
            ),
        }
        estimates[name] = estimate
        models[name] = model
    save(
        "wang-ptbxl-lead-morphology" if real else "wang-latent-inverse-leads",
        real,
        {
            "source": "PTB-XL+ GE 12SL measurements"
            if real
            else "simulated nonlinear source and torso variation",
            "training_records": len(train),
            "test_records": len(test),
            "metrics": report,
        },
        {"models": models, "input_scaler": p, "target_scaler": q},
        {
            "test_ids": ids[test],
            "train_ids": ids[train],
            "scaled_targets": v[test],
            **estimates,
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            run(real)
