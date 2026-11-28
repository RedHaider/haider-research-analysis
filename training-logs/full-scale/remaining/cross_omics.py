import numpy as np
import pandas as pd
from common import DATA, save
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def data(real):
    if real:
        mirna = pd.read_csv(DATA / "paad-mirna.tsv.gz", sep="\t", index_col=0).T
        mrna = pd.read_csv(DATA / "paad-mrna.tsv.gz", sep="\t", index_col=0).T
        ids = sorted(
            x
            for x in set(mirna.index) & set(mrna.index)
            if str(x).split("-")[-1].startswith("01")
        )
        assert len(ids) > 100 and len({i[:12] for i in ids}) == len(ids)
        return (
            mrna.loc[ids].to_numpy(dtype=np.float32),
            mirna.loc[ids].to_numpy(dtype=np.float32),
            np.array(ids),
        )
    r = np.random.default_rng(36)
    base = r.normal(size=(1400, 10))
    return (
        (base @ r.normal(size=(10, 640)) + r.normal(size=(1400, 640))).astype(
            np.float32
        ),
        (base @ r.normal(size=(10, 48)) + r.normal(size=(1400, 48))).astype(np.float32),
        np.arange(1400),
    )


def columns(x, train, n):
    fit = x[train]
    finite = np.isfinite(fit).mean(0) > 0.94
    visible = np.flatnonzero(finite)
    var = np.nanvar(fit[:, visible], axis=0)
    return visible[np.argsort(var)[-n:]]


def main(real):
    X, Y, ids = data(real)
    order = np.random.default_rng(36).permutation(len(X))
    cut = int(0.78 * len(X))
    tr, te = order[:cut], order[cut:]
    choose, target = columns(X, tr, 256), columns(Y, tr, 32)
    X, Y = X[:, choose], Y[:, target]
    for table in (X, Y):
        center = np.nanmedian(table[tr], axis=0)
        missing = np.where(~np.isfinite(table))
        table[missing] = center[missing[1]]
    sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
    X, Y = sx.transform(X), sy.transform(Y)
    report, prediction, models = {}, {}, {}
    for name, method in (
        ("ridge", Ridge(alpha=45)),
        ("shared_latent_pls", PLSRegression(n_components=12, scale=False)),
    ):
        method.fit(X[tr], Y[tr])
        p = method.predict(X[te])
        corrs = [
            np.corrcoef(p[:, i], Y[te, i])[0, 1]
            for i in range(len(target))
            if np.std(p[:, i]) > 0 and np.std(Y[te, i]) > 0
        ]
        report[name] = {
            "rmse": float(np.sqrt(np.mean((p - Y[te]) ** 2))),
            "mean_target_correlation": float(np.mean(corrs)),
        }
        prediction[name], models[name] = p, method
    save(
        "cui-paad-cross-omics" if real else "cui-mirna-integrative-network",
        real,
        {"records": len(X), "test_records": len(te), "metrics": report},
        {
            "models": models,
            "x_scaler": sx,
            "y_scaler": sy,
            "input_columns": choose,
            "target_columns": target,
        },
        {
            "targets": Y[te],
            "training_cases": ids[tr],
            "test_cases": ids[te],
            **prediction,
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
