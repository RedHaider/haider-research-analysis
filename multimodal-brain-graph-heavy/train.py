"""Module-aware structural and functional edge fusion on simulated paired graphs."""

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N = 24
I, J = np.triu_indices(N, 1)
MODULE = np.arange(N) // 6
BLOCKS = [(MODULE[I] == a) & (MODULE[J] == b) for a in range(4) for b in range(a, 4)]


def generate(n, seed, shift):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    labels = (
        latent[:, 0] + latent[:, 1] + 0.8 * latent.prod(1) + rng.normal(0, 0.65, n) > 0
    )
    struct = []
    func = []
    for u, v in latent:
        rate = np.full(len(I), 0.9) + 0.85 * (MODULE[I] == MODULE[J])
        rate[BLOCKS[0]] += 0.40 * np.tanh(u)
        a = rng.poisson(rate * np.exp(rng.normal(0, 0.14) + 0.18 * shift))
        shared = rng.normal(size=(72, 4))
        load = np.full(N, 0.65)
        load[MODULE == 2] += 0.33 * np.tanh(v)
        ts = shared[:, MODULE] * load + rng.normal(size=(72, N)) * (1 + 0.18 * shift)
        struct.append(np.log1p(a))
        func.append(np.arctanh(np.clip(np.corrcoef(ts.T)[I, J], -0.995, 0.995)))
    return np.array(struct), np.array(func), labels


def features(s, f):
    def pool(x):
        return np.stack([x[:, b].mean(1) for b in BLOCKS], 1)

    a, b = pool(s), pool(f)
    interactions = (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)
    return {
        "structural": s,
        "functional": f,
        "edge_concat": np.c_[s, f],
        "pooled": np.c_[a, b],
        "pooled_interactions": np.c_[a, b, interactions],
    }


def run(seed):
    splits = [
        generate(n, seed + offset, shift)
        for n, offset, shift in [(2500, 0, 0), (600, 1, 0), (900, 2, 0), (900, 2, 1)]
    ]
    inputs = [features(s, f) for s, f, _ in splits]
    result = {}
    for method in inputs[0]:
        candidates = []
        for c in [0.01, 0.1, 1.0]:
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=c, max_iter=400)
            )
            model.fit(inputs[0][method], splits[0][2])
            v = model.predict_proba(inputs[1][method])[:, 1]
            candidates.append((brier_score_loss(splits[1][2], v), model, c))
        _, model, c = min(candidates, key=lambda t: t[0])
        dest = Path(__file__).with_name("checkpoints")
        dest.mkdir(exist_ok=True)
        scaler, lr = model.steps[0][1], model.steps[1][1]
        np.savez_compressed(
            dest / f"{method}-{seed}.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            coef=lr.coef_,
            intercept=lr.intercept_,
        )
        for k, tag in [(2, "clean"), (3, "shifted")]:
            p = model.predict_proba(inputs[k][method])[:, 1]
            result[method + "_" + tag] = {
                "auroc": roc_auc_score(splits[k][2], p),
                "brier": brier_score_loss(splits[k][2], p),
                "C": c,
            }
    return result


if __name__ == "__main__":
    runs = {s: run(s) for s in [17, 29, 43, 59, 71]}
    means = {
        k: {m: np.mean([r[k][m] for r in runs.values()]) for m in ["auroc", "brier"]}
        for k in next(iter(runs.values()))
    }
    Path(__file__).with_name("results.json").write_text(
        json.dumps({"runs": runs, "means": means}, indent=2)
    )
