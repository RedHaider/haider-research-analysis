from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
from common import DATA, classification, save, split
from scipy import signal
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def measured():
    pieces, targets, people = [], [], []
    with ZipFile(DATA / "emg-gestures.zip") as z:
        for name in sorted(z.namelist()):
            parent = PurePosixPath(name).parent.name
            if not name.endswith(".txt") or not parent.isdigit():
                continue
            with z.open(name) as stream:
                next(stream)
                rows = (
                    line if len(line.split()) == 10 else b"0 0 0 0 0 0 0 0 0 0"
                    for line in stream
                )
                table = np.loadtxt(rows, dtype=np.float32)
            y = table[:, -1].astype(int)
            at = np.arange(0, len(y) - 252, 125)
            inside = np.arange(0, 250, 3) + at[:, None]
            choose = (
                (y[inside] == y[inside[:, 0], None]).all(1)
                & (y[at] >= 1)
                & (y[at] <= 6)
            )
            pieces.append(table[inside[choose], 1:9])
            targets.append(y[at[choose]])
            people.append(np.full(choose.sum(), int(parent)))
    return np.concatenate(pieces), np.concatenate(targets), np.concatenate(people)


def simulate():
    rng = np.random.default_rng(82)
    people = np.repeat(np.arange(96), 108)
    classes = np.tile(np.arange(108) % 6 + 1, 96)
    channels = rng.uniform(0.3, 1, (6, 8))
    gain = rng.uniform(0.65, 1.4, (96, 8))
    time = np.arange(84) / 333
    x = rng.normal(size=(len(people), 84, 8)) * 0.35
    x += (
        np.sin(
            2 * np.pi * (20 + classes[:, None, None] * 8) * time[None, :, None]
            + rng.uniform(0, 6, (len(people), 1, 8))
        )
        * channels[classes - 1, None, :]
    )
    return (x * gain[people, None, :]).astype(np.float32), classes, people


def run(real):
    x, y, groups = measured() if real else simulate()
    train, test = split(y, groups)
    time = np.column_stack(
        [
            np.sqrt((x**2).mean(1)),
            abs(x).mean(1),
            abs(np.diff(x, axis=1)).mean(1),
            np.signbit(x[:, 1:] * x[:, :-1]).mean(1),
        ]
    )
    _, _, z = signal.stft(x, fs=333.33, nperseg=24, noverlap=12, axis=1)
    spectrum = abs(z) ** 2
    tf = np.column_stack(
        [
            np.log1p(p.mean(axis=(1, 3)) * 1e5)
            for p in np.array_split(spectrum, 5, axis=1)
        ]
    )
    results, models, predictions = {}, {}, {}
    for name, features in (
        ("time_only", time),
        ("time_frequency", np.column_stack([time, tf])),
    ):
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(192, 96),
                max_iter=45,
                early_stopping=True,
                batch_size=128,
                random_state=82,
            ),
        ).fit(features[train], y[train])
        score = clf.predict_proba(features[test])
        results[name] = classification(y[test], score, clf.classes_)
        models[name], predictions[name] = clf, score
    save(
        "rattani-uci-emg-gestures" if real else "rattani-emg-time-frequency",
        real,
        {
            "windows": len(y),
            "subjects": len(np.unique(groups)),
            "test_subjects": len(np.unique(groups[test])),
            "metrics": results,
        },
        models,
        {
            "test_labels": y[test],
            "train_groups": groups[train],
            "test_groups": groups[test],
            **predictions,
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            run(real)
