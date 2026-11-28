import tarfile

import numpy as np
from common import DATA, save
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def read_real():
    base = "data/networks/human/human_string_"
    ann = "data/annotations/human/go_human_ref_"
    with tarfile.open(DATA / "mashup.tar.gz") as t:

        def genes(path):
            return t.extractfile(path).read().decode().split()

        def matrix(path):
            return np.loadtxt(t.extractfile(path))

        ids = np.array(genes(base + "genes.txt"))
        codes = {name: i for i, name in enumerate(ids)}
        annotation = genes(ann + "genes.txt")
        rows = matrix(ann + "cc_adjacency.txt").astype(int) - 1
        target = np.zeros((len(ids), rows[:, 1].max() + 1), dtype=np.uint8)
        for source, term in rows:
            found = codes.get(annotation[source])
            if found is not None:
                target[found, term] = 1
        graphs = []
        for kind in ("experimental", "coexpression", "cooccurence"):
            data = matrix(base + kind + "_adjacency.txt")
            n = data[:, :2].astype(np.int32) - 1
            adj = sparse.csr_matrix(
                (data[:, 2].astype(np.float32), (n[:, 0], n[:, 1])),
                shape=(len(ids), len(ids)),
            )
            graphs.append(adj.maximum(adj.T))
    return ids, graphs, target


def simulate():
    rng = np.random.default_rng(22)
    n = 15000
    labels = np.arange(n) % 12
    graphs = []
    for channel in range(3):
        u = np.repeat(np.arange(n), 10)
        v = rng.integers(n, size=len(u))
        retained = rng.random(len(u)) < (0.52 + 0.08 * channel)
        v[retained] = (
            rng.integers(n // 12, size=retained.sum()) * 12 + labels[u[retained]]
        )
        adj = sparse.csr_matrix(
            (np.ones(len(u), dtype=np.float32), (u, v)), shape=(n, n)
        )
        graphs.append(adj.maximum(adj.T))
    return np.arange(n), graphs, np.eye(12, dtype=np.uint8)[labels]


def features(A, anchors):
    degree = np.asarray(A.sum(1)).ravel()
    norm = A.multiply((1 / np.maximum(degree, 1e-6))[:, None]).tocsr()
    z = anchors
    result = []
    for _ in range(4):
        z = 0.12 * anchors + 0.88 * (norm @ z)
        result.append(z)
    return np.column_stack([*result, np.log1p(degree)])


def run(real):
    ids, graphs, y = read_real() if real else simulate()
    rng = np.random.default_rng(22)
    shuffled = rng.permutation(len(ids))
    cut = int(0.78 * len(ids))
    train, test = shuffled[:cut], shuffled[cut:]
    terms = np.argsort(y[train].sum(0))[-12:]
    y = y[:, terms]
    test_viable = y[test].sum(0) > 0
    assert test_viable.all()
    anchor = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), (len(ids), 32))
    arms = {}
    measured = {}
    predictions = {}
    for name, raw in (
        ("pooled", features(sum(graphs), anchor)),
        ("network_specific", np.column_stack([features(a, anchor) for a in graphs])),
    ):
        norm = StandardScaler().fit(raw[train])
        a = norm.transform(raw)
        compact = PCA(n_components=80, svd_solver="randomized", random_state=22).fit(
            a[train]
        )
        a = compact.transform(a)
        models = [
            LogisticRegression(C=0.5, max_iter=250, class_weight="balanced").fit(
                a[train], y[train, j]
            )
            for j in range(y.shape[1])
        ]
        p = np.column_stack([model.predict_proba(a[test])[:, 1] for model in models])
        measured[name] = {
            "macro_ap": float(average_precision_score(y[test], p, average="macro")),
            "micro_ap": float(average_precision_score(y[test], p, average="micro")),
        }
        predictions[name] = p
        arms[name] = (norm, compact, models)
    save(
        "peng-human-network-integration" if real else "peng-multiplex-diffusion",
        real,
        {
            "nodes": len(ids),
            "networks": len(graphs),
            "training_nodes": len(train),
            "test_nodes": len(test),
            "annotation_terms": terms.tolist(),
            "metrics": measured,
        },
        arms,
        {
            "train_ids": ids[train],
            "test_ids": ids[test],
            "labels": y[test],
            **predictions,
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            run(real)
