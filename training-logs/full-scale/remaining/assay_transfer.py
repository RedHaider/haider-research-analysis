import io
import tarfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from common import DATA, classification, save
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def source(real):
    if real:
        with (
            ZipFile(DATA / "tcga-rnaseq.zip") as z,
            tarfile.open(fileobj=z.open(z.namelist()[0]), mode="r|gz") as t,
        ):
            files = {}
            for member in t:
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                part = pd.read_csv(
                    io.BytesIO(t.extractfile(member).read()), index_col=0
                )
                files[member.name.rsplit("/", 1)[-1]] = part
        merged = files["data.csv"].join(files["labels.csv"], validate="one_to_one")
        return (
            merged.drop(columns="Class").to_numpy(dtype=np.float32),
            merged.Class.to_numpy(),
            merged.index.to_numpy(),
        )
    rng = np.random.default_rng(41)
    y = np.arange(5600) % 5
    a = rng.normal(0, 0.09, (5, 768))
    hidden = rng.normal(0, 0.8, (len(y), 12)) @ rng.normal(0, 0.22, (12, 768))
    return (
        (a[y] + hidden + rng.normal(0, 1, hidden.shape)).astype(np.float32),
        y,
        np.arange(len(y)),
    )


def main(real):
    x, y, ids = source(real)
    train, rest = train_test_split(
        np.arange(len(y)), test_size=0.46, stratify=y, random_state=41
    )
    adapt, test = train_test_split(
        rest, test_size=0.55, stratify=y[rest], random_state=42
    )
    features = np.argsort(x[train].var(0))[-768:]
    center = StandardScaler().fit(x[train][:, features])
    x = center.transform(x[:, features]).clip(-7, 7)
    r = np.random.default_rng(41)
    shifted = (
        x * r.uniform(0.30, 1.85, x.shape[1])
        + r.normal(0, 0.9, x.shape[1])
        + r.normal(0, 0.22, x.shape)
    )
    reducer = PCA(n_components=60, svd_solver="randomized", random_state=41).fit(
        x[train]
    )
    z = reducer.transform(x[train])
    w = reducer.transform(shifted[adapt])
    v = reducer.transform(shifted[test])
    ms = z.mean(0)
    mt = w.mean(0)

    def root(a, power):
        val, vec = np.linalg.eigh(np.cov(a, rowvar=False) + np.eye(a.shape[1]) * 0.14)
        return (vec * val**power) @ vec.T

    transfer = root(w, -0.5) @ root(z, 0.5)
    wa, va = (w - mt) @ transfer + ms, (v - mt) @ transfer + ms
    base = LogisticRegression(C=1, max_iter=500).fit(z, y[train])
    confidence = base.predict_proba(wa)
    selected = confidence.max(1) >= 0.75
    labels = base.classes_[confidence.argmax(1)]
    adapted = LogisticRegression(C=1, max_iter=500).fit(
        np.concatenate((z, wa[selected])), np.r_[y[train], labels[selected]]
    )
    p = {
        "source_only": base.predict_proba(v),
        "moment_alignment": base.predict_proba(va),
        "semi_supervised": adapted.predict_proba(va),
    }
    save(
        "zhang-tcga-assay-shift" if real else "zhang-semi-supervised-assay-transfer",
        real,
        {
            "samples": len(y),
            "training": len(train),
            "unlabelled_adaptation": len(adapt),
            "test": len(test),
            "shift": "artificial affine noise on distinct target rows",
            "pseudo_labels": int(selected.sum()),
            "metrics": {
                n: classification(y[test], q, base.classes_) for n, q in p.items()
            },
        },
        {
            "baseline": base,
            "adapted": adapted,
            "pca": reducer,
            "scaler": center,
            "source_mean": ms,
            "target_mean": mt,
            "transport": transfer,
            "features": features,
        },
        {
            "labels": y[test],
            "training_ids": ids[train],
            "adaptation_ids": ids[adapt],
            "test_ids": ids[test],
            **p,
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
