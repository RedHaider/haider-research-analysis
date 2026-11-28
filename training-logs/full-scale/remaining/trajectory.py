import io
from zipfile import ZipFile

import numpy as np
from common import DATA, classification, save
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def har():
    parts = []
    with (
        ZipFile(DATA / "har.zip") as outer,
        ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as z,
    ):
        for mode in ("train", "test"):

            def read(f, mode=mode):
                with z.open(f"UCI HAR Dataset/{mode}/{f}_{mode}.txt") as file:
                    return np.loadtxt(file, dtype=np.float32)

            parts.append(
                (read("X"), read("y").astype(int) - 1, read("subject").astype(int))
            )
    assert not set(parts[0][2]) & set(parts[1][2])
    return parts


def simulate():
    rng = np.random.default_rng(75)
    signatures = rng.normal(0, 0.34, (6, 48))

    def part(n, offset):
        x, y, s = [], [], []
        for subject in range(offset, offset + n):
            path = (np.arange(240) // rng.integers(7, 21) + rng.integers(6)) % 6
            x.append(
                signatures[path]
                + rng.normal(0, 0.93, (len(path), 48))
                + rng.normal(0, 0.15, (1, 48))
            )
            y.extend(path)
            s.extend([subject] * len(path))
        return np.concatenate(x).astype(np.float32), np.asarray(y), np.asarray(s)

    return [part(65, 0), part(20, 65)]


def transition(y, subjects, k):
    counts = np.full((k, k), 2.0)
    for sub in np.unique(subjects):
        values = y[subjects == sub]
        np.add.at(counts, (values[:-1], values[1:]), 1)
    return np.log(counts / counts.sum(1, keepdims=True))


def decode(logp, A, groups, strength):
    predicted = np.empty(len(groups), dtype=int)
    for sub in np.unique(groups):
        rows = np.flatnonzero(groups == sub)
        scores = logp[rows[0]].copy()
        back = np.zeros((len(rows), len(scores)), dtype=int)
        for i, idx in enumerate(rows[1:], 1):
            values = scores[:, None] + strength * A
            back[i] = values.argmax(0)
            scores = logp[idx] + values.max(0)
        predicted[rows[-1]] = scores.argmax()
        for i in range(len(rows) - 1, 0, -1):
            predicted[rows[i - 1]] = back[i, predicted[rows[i]]]
    return predicted


def run(real):
    (x, y, g), (xx, yy, gg) = har() if real else simulate()
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(192, 128),
            max_iter=50,
            early_stopping=True,
            batch_size=192,
            random_state=75,
        ),
    )
    clf.fit(x, y)
    scores = clf.predict_proba(xx)
    A = transition(y, g, len(clf.classes_))
    p = scores.argmax(1)
    q = decode(np.log(scores.clip(1e-8, 1)), A, gg, 0.4)
    save(
        "liu-har-contextual-decoding" if real else "liu-trajectory-context-graph",
        real,
        {
            "train_windows": len(y),
            "test_windows": len(yy),
            "test_subjects": len(np.unique(gg)),
            "metrics": {
                "local": classification(yy, p),
                "trajectory": classification(yy, q),
            },
            "transition_strength": 0.4,
        },
        {"emitter": clf, "transition": A},
        {"subject": gg, "truth": yy, "local": p, "trajectory": q},
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            run(real)
