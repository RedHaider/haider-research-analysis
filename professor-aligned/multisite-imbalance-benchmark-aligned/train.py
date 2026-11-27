import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PORTFOLIO_DATA", ROOT / "data"))
SEEDS = [13, 37, 61]


def save(obj, name):
    folder = ROOT / "checkpoints"
    folder.mkdir(exist_ok=True)
    joblib.dump(obj, folder / (name + ".joblib"), compress=3)


def scores(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


from scipy.special import expit


def dataset(seed):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    sites = []
    for site, n, bias in [
        (0, 700, -2.0),
        (1, 350, -1.5),
        (2, 190, -0.2),
        (3, 110, 0.7),
    ]:
        x = rng.normal(size=(n, 12))
        x[:, 1] += 0.5 * site
        p = expit(
            bias + 1.1 * x[:, 0] + (0.55 - 0.35 * site) * x[:, 1] + 0.45 * x[:, 3]
        )
        y = rng.binomial(1, p)
        xs.append(x)
        ys.extend(y)
        sites.extend([site] * n)
    return np.vstack(xs), np.asarray(ys), np.asarray(sites)


from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight


def run(seed):
    x, y, site = dataset(seed)
    result = {}
    for scheme in ["pooled", "site_class_weighted"]:
        site_results = []
        for held in np.unique(site):
            tr = np.where(site != held)[0]
            te = np.where(site == held)[0]
            rng = np.random.default_rng(seed)
            tr = np.concatenate(
                [
                    rng.choice(sub, max(8, int(len(sub) * 0.9)), replace=False)
                    for g in np.unique(site[tr])
                    for sub in [tr[site[tr] == g]]
                ]
            )
            risk = make_pipeline(
                SimpleImputer(add_indicator=True),
                StandardScaler(),
                LogisticRegression(C=0.12, max_iter=600),
            )
            w = np.ones(len(tr))
            if scheme == "site_class_weighted":
                for g in np.unique(site[tr]):
                    use = site[tr] == g
                    w[use] = compute_sample_weight("balanced", y[tr][use]) / use.sum()
                w *= len(w) / w.sum()
            risk.fit(x[tr], y[tr], logisticregression__sample_weight=w)
            p = risk.predict_proba(x[te])[:, 1]
            site_results.append(
                {**scores(y[te], p), "brier": float(brier_score_loss(y[te], p))}
            )
            save(risk, f"{scheme}-{held}-{seed}")
        result[scheme] = {
            k: float(np.mean([a[k] for a in site_results])) for k in site_results[0]
        }
        result[scheme]["worst_site_auroc"] = min(a["auroc"] for a in site_results)
    return result


if __name__ == "__main__":
    runs = {s: run(s) for s in SEEDS}
    first = next(iter(runs.values()))
    means = {
        k: {m: float(np.mean([r[k][m] for r in runs.values()])) for m in v}
        for k, v in first.items()
    }
    (ROOT / "results.json").write_text(
        json.dumps({"seeds": SEEDS, "runs": runs, "means": means}, indent=2)
    )
    print(means)
