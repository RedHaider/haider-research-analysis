import numpy as np
import pandas as pd
from common import DATA, classification, save
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


def source(real):
    if real:
        meta = pd.read_csv(DATA / "METABRIC_metadata.tsv", sep="\t")
        expression = pd.read_csv(DATA / "METABRIC.tsv.gz", sep="\t", low_memory=False)
        months = pd.to_numeric(meta.OS_MONTHS, errors="coerce")
        dead = meta.OS_STATUS.astype(str).str.lower().str.contains("deceas|^1:")
        ok = (months >= 60) | (dead & months.notna())
        keep = meta.loc[ok & meta.Sample_ID.isin(expression.columns)].drop_duplicates(
            "Sample_ID"
        )
        columns = [
            "AGE_AT_DIAGNOSIS",
            "NPI",
            "LYMPH_NODES_EXAMINED_POSITIVE",
            "CELLULARITY",
            "ER_IHC",
            "HISTOLOGICAL_SUBTYPE",
        ]
        y = (dead & (months < 60)).loc[keep.index].astype(int).to_numpy()
        return (
            keep[columns].copy(),
            expression[keep.Sample_ID.tolist()].to_numpy(dtype=np.float32).T,
            y,
            keep.COHORT.to_numpy(int),
            keep.Sample_ID.to_numpy(),
        )
    r = np.random.default_rng(72)
    n = 4200
    signals = r.normal(size=(n, 12))
    y = (
        signals[:, 0] * signals[:, 3]
        + signals[:, 1] * 0.9
        + signals[:, 9] * 1.2
        + r.normal(0, 1.2, n)
        > 0
    ).astype(int)
    clinic = pd.DataFrame(
        signals[:, :6] + r.normal(0, 0.36, (n, 6)), columns=[f"v{i}" for i in range(6)]
    )
    expression = signals[:, 6:] @ r.normal(0, 0.45, (6, 480)) + r.normal(0, 1, (n, 480))
    return clinic, expression, y, np.arange(n) % 5, np.arange(n)


def main(real):
    clinic, expression, y, cohort, ids = source(real)
    test = np.flatnonzero(cohort == (3 if real else 4))
    tr = np.flatnonzero(cohort != (3 if real else 4))
    assert len(np.unique(y[test])) == 2
    numeric = clinic.select_dtypes(include="number").columns
    categorical = clinic.columns.difference(numeric)
    encode = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(numeric),
            ),
            (
                "category",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                list(categorical),
            ),
        ]
    )
    clinical = encode.fit(clinic.iloc[tr]).transform(clinic).astype(np.float32)
    good = np.flatnonzero(np.isfinite(expression[tr]).mean(0) > 0.98)
    features = good[np.argsort(np.nanvar(expression[tr][:, good], axis=0))[-256:]]
    expression = expression[:, features]
    norm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    ex = norm.fit(expression[tr]).transform(expression).clip(-8, 8)
    pred, result, models = {}, {}, {}
    for name, x in (
        ("clinical", clinical),
        ("expression", ex),
        ("fused", np.column_stack([clinical, ex])),
    ):
        model = MLPClassifier(
            hidden_layer_sizes=(192, 96),
            batch_size=128,
            alpha=0.02,
            max_iter=70,
            early_stopping=True,
            random_state=72,
        ).fit(x[tr], y[tr])
        p = model.predict_proba(x[test])
        models[name], pred[name] = model, p
        result[name] = {
            **classification(y[test], p, model.classes_),
            "ap": float(average_precision_score(y[test], p[:, 1])),
        }
    save(
        "singhal-metabric-risk-fusion" if real else "singhal-multimodal-risk-fusion",
        real,
        {
            "label": "death within five years, excluding early-censored records"
            if real
            else "simulated risk",
            "training_records": len(tr),
            "test_records": len(test),
            "test_cohort": int(cohort[test[0]]),
            "event_fraction": float(y[test].mean()),
            "metrics": result,
        },
        {
            "models": models,
            "clinical_preprocessor": encode,
            "expression_preprocessor": norm,
            "expression_columns": features,
        },
        {"training_ids": ids[tr], "test_ids": ids[test], "target": y[test], **pred},
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
