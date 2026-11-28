import gzip

import numpy as np
import pandas as pd
from common import DATA, classification, save
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

GENES = {
    "CD3D",
    "CD3E",
    "CD8A",
    "CD8B",
    "CD4",
    "TCF7",
    "LEF1",
    "CCR7",
    "IL7R",
    "GZMB",
    "GZMK",
    "PRF1",
    "NKG7",
    "PDCD1",
    "HAVCR2",
    "LAG3",
    "TIGIT",
    "CTLA4",
    "TOX",
    "CXCR6",
    "CX3CR1",
    "IFNG",
    "CXCL13",
    "CD69",
    "ITGAE",
    "HLA-DRA",
    "MKI67",
    "FOS",
    "JUN",
    "MS4A1",
    "LYZ",
    "CD14",
    "CD68",
    "TRAC",
    "CD27",
    "SELL",
}


def measured():
    with gzip.open(DATA / "melanoma-expression.tsv.gz", "rt") as file:
        cells = file.readline().rstrip("\n\r").split("\t")[1:]
        file.readline()
        block = {}
        for line in file:
            key, _, tail = line.partition("\t")
            if key in GENES:
                block[key] = np.fromstring(tail.rstrip(), sep="\t", dtype=np.float32)
    mat = np.column_stack([block[n] for n in sorted(block)])
    assert len(cells) == len(mat)
    with gzip.open(DATA / "melanoma-metadata.tsv.gz", "rt") as f:
        tab = pd.read_csv(f, sep="\t", skiprows=19, low_memory=False)
    patient_col = next(k for k in tab if "patinet ID" in k)
    response = next(k for k in tab if "characteristics: response" in k)
    meta = tab.drop_duplicates("title").set_index("title").reindex(cells)
    state = meta[patient_col].fillna("")
    keep = state.str.startswith("Pre_") & meta[response].isin(
        ["Responder", "Non-responder"]
    )
    if keep.sum() < 1000:
        expected = (
            meta[response]
            .fillna("")
            .str.lower()
            .str.replace(" ", "")
            .str.replace("-", "")
        )
        keep = state.str.startswith("Pre_") & expected.isin(
            ["responder", "nonresponder"]
        )
        labels = (expected == "responder").astype(int)
    else:
        labels = (meta[response] == "Responder").astype(int)
    assert keep.sum() > 1000
    return (
        np.log1p(mat[keep]),
        labels.to_numpy()[keep],
        state.to_numpy()[keep],
        len(cells),
    )


def simulate():
    r = np.random.default_rng(43)
    g = np.repeat(np.arange(160), 112)
    y = g % 2
    x = r.normal(0, 1, (len(g), 32)) + r.normal(0, 0.4, (160, 32))[g]
    state = r.random(len(y)) < np.where(y == 1, 0.3, 0.1)
    x[state, :8] += 2.1
    return x, y, g, len(x)


def main(real):
    x, y, g, available = measured() if real else simulate()
    means, dist, outcomes = [], [], []
    subjects = np.unique(g)
    for who in subjects:
        subset = g == who
        cells = x[subset]
        labels = y[subset]
        assert len(np.unique(labels)) == 1
        typical = cells.mean(0)
        means.append(typical)
        dist.append(
            np.r_[
                typical,
                np.quantile(cells, 0.8, axis=0),
                np.quantile(cells, 0.95, axis=0),
                cells.std(0),
            ]
        )
        outcomes.append(labels[0])
    truth = np.asarray(outcomes)
    folds = StratifiedKFold(
        n_splits=min(5, int(np.bincount(truth).min())), shuffle=True, random_state=43
    )
    methods, result, models = {}, {}, {}
    for name, z in (
        ("mean", np.asarray(means)),
        ("state_distributions", np.asarray(dist)),
    ):
        score = np.zeros((len(subjects), 2))
        models[name] = []
        for train, test in folds.split(z, truth):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.055, max_iter=500, class_weight="balanced"),
            ).fit(z[train], truth[train])
            score[test] = model.predict_proba(z[test])
            models[name].append((model, subjects[test]))
        result[name] = classification(truth, score, np.array([0, 1]))
        methods[name] = score
    save(
        "pritykin-melanoma-cell-states" if real else "pritykin-cell-mixture-states",
        real,
        {
            "profiled_cells": available,
            "analysed_baseline_cells": len(y),
            "patients": len(subjects),
            "held_out_method": "patient-stratified out-of-fold",
            "metrics": result,
        },
        models,
        {"patients": subjects, "reference": truth, **methods},
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
