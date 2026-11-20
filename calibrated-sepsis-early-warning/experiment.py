import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def run(seed=42):
    rng = np.random.default_rng(seed)
    n = 2200
    x = rng.normal(size=(n, 10))
    interaction = x[:, 0] * x[:, 1]
    probability = 1 / (1 + np.exp(-(0.9*x[:,0] - 0.7*x[:,2] + 0.65*interaction)))
    y = rng.binomial(1, probability)
    split = 1500
    feature_sets = {"baseline": x, "enriched": np.column_stack([x, interaction])}
    result = {}
    for name, features in feature_sets.items():
        model = LogisticRegression(max_iter=500).fit(features[:split], y[:split])
        score = model.predict_proba(features[split:])[:, 1]
        result[name] = {"auroc": float(roc_auc_score(y[split:], score))}
    os.makedirs("outputs/demo", exist_ok=True)
    with open("outputs/demo/metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

