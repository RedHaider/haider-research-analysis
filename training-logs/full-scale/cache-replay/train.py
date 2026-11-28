"""Replay two-level cache traces with learned, cost-aware admission."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
LOOKAHEAD = 4096


class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()
        self.loaded = 0

    def remove(self, key):
        size = self.items.pop(key, 0)
        self.loaded -= size
        return size

    def insert(self, key, size):
        self.remove(key)
        victims = []
        if size > self.capacity:
            return victims
        while self.loaded + size > self.capacity:
            victim, size_victim = self.items.popitem(last=False)
            self.loaded -= size_victim
            victims.append((victim, size_victim))
        self.items[key] = size
        self.loaded += size
        return victims

    def hit(self, key):
        if key not in self.items:
            return False
        self.items.move_to_end(key)
        return True


def public_trace(cluster, limit):
    import zstandard

    path = ROOT / "datasets" / f"twitter-cluster{cluster}-prefix.zst"
    dtype = np.dtype(
        [
            ("timestamp", "<u4"),
            ("object", "<u8"),
            ("size", "<u4"),
            ("future_unused", "<i8"),
        ]
    )
    pieces = []
    with (
        path.open("rb") as file,
        zstandard.ZstdDecompressor().stream_reader(file) as stream,
    ):
        while (missing := limit * dtype.itemsize - sum(map(len, pieces))) > 0:
            data = stream.read(min(2**21, missing))
            if not data:
                break
            pieces.append(data)
    available = b"".join(pieces)
    rows = np.frombuffer(
        available[: (len(available) // dtype.itemsize) * dtype.itemsize], dtype=dtype
    )
    if len(rows) < 100000 or (np.diff(rows["timestamp"].astype(np.int64)) < 0).any():
        raise ValueError("Unexpected data order or trace length")
    keep = rows["size"] > 0
    _, objects = np.unique(rows["object"][keep], return_inverse=True)
    return objects.astype(np.int32), rows["size"][keep].clip(16, 65536).astype(np.int32)


def generate(seed, num):
    rng = np.random.default_rng(seed)
    n = 32000
    raw = np.empty(num, dtype=np.int32)
    hot = rng.permutation(n)
    weights = (np.arange(n) + 1.0) ** -0.95
    weights /= weights.sum()
    for phase, first in enumerate(range(0, num, 20000)):
        end = min(num, first + 20000)
        choices = rng.choice(n, size=end - first, p=weights)
        raw[first:end] = hot[(choices + (phase % 6) * 31) % n]
    size = np.clip(np.exp(rng.normal(6.5, 1.45, size=n)), 16, 65536).astype(np.int32)
    return raw, size[raw]


def feature_history(keys, sizes):
    nkeys = int(keys.max()) + 1
    count = np.zeros(nkeys, dtype=np.int32)
    last = np.full(nkeys, -1, dtype=np.int32)
    features = np.empty((len(keys), 4), dtype=np.float32)
    for i, (key, size) in enumerate(zip(keys, sizes)):
        gap = i - last[key] if count[key] else len(keys)
        features[i] = (
            np.log1p(size),
            np.log1p(count[key]),
            np.log1p(gap),
            count[key] > 0,
        )
        last[key] = i
        count[key] += 1
    return features


def candidate_features(features, last_access, which, step):
    reference = last_access[which]
    return (*features[reference], np.log1p(step - reference))


def next_indices(keys):
    upcoming = np.full(int(keys.max()) + 1, -1, dtype=np.int32)
    result = np.empty(len(keys), dtype=np.int32)
    for i in range(len(keys) - 1, -1, -1):
        result[i] = upcoming[keys[i]]
        upcoming[keys[i]] = i
    return result


def make_training_rows(trace, capacities, warmup):
    keys, sizes = trace
    features = feature_history(keys, sizes)
    future = next_indices(keys)
    last = np.full(int(keys.max()) + 1, -1, dtype=np.int32)
    first = LRU(capacities[0])
    x, y = [], []
    for t, (key, size) in enumerate(zip(keys, sizes)):
        last[key] = t
        if first.hit(key):
            continue
        for victim, _ in first.insert(key, int(size)):
            if warmup <= t < len(keys) - LOOKAHEAD:
                following = future[last[victim]]
                x.append(candidate_features(features, last, victim, t))
                y.append(t < following <= t + LOOKAHEAD)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=int)


def scores_lookup(model, feature_rows, seed, seed_limit=50000):
    # First-order feature binning builds a bounded table for fast inference during replay.
    anchors = feature_rows[
        np.random.default_rng(seed).choice(
            len(feature_rows), min(seed_limit, len(feature_rows)), replace=False
        )
    ]
    boundaries = [
        np.unique(np.quantile(anchors[:, c], np.linspace(0.05, 0.95, 12)))
        for c in range(5)
    ]
    centers = []
    for c, cuts in enumerate(boundaries):
        extended = np.r_[anchors[:, c].min(), cuts, anchors[:, c].max()]
        centers.append((extended[:-1] + extended[1:]) / 2)
    grid = np.stack([v.ravel() for v in np.meshgrid(*centers, indexing="ij")], axis=1)
    lookup = model.predict_proba(grid)[:, 1].reshape(*map(len, centers))
    return boundaries, lookup


def replay(trace, capacities, warmup, policy, lookup=None, ref=None, threshold=0.5):
    key_seq, size_seq = trace
    last = np.full(int(key_seq.max()) + 1, -1, dtype=np.int32)
    features = feature_history(key_seq, size_seq) if lookup is not None else None
    first, second = [LRU(capacity) for capacity in capacities]
    counters = np.zeros(5, dtype=np.int64)
    for t, (key, size) in enumerate(zip(key_seq, size_seq)):
        last[key] = t
        count = t >= warmup
        if first.hit(key):
            counters[0] += count
            continue
        found = second.remove(key)
        counters[1] += count and found > 0
        if count and not found:
            counters[2] += 1
            counters[3] += size
        for victim, released in first.insert(key, int(size)):
            if policy == "second_touch" and ref[last[victim], 3] == 0:
                continue
            if policy == "learned":
                bounds, grid = lookup
                candidate = candidate_features(features, last, victim, t)
                bins = tuple(
                    np.searchsorted(cuts, value, side="right")
                    for cuts, value in zip(bounds, candidate)
                )
                if grid[bins] < threshold:
                    continue
            if released <= second.capacity:
                second.insert(victim, released)
                counters[4] += released * count
    requests = len(key_seq) - warmup
    capacity = sum(capacities)
    return {
        "capacity_bytes": capacity,
        "requests": requests,
        "l1_hit_rate": float(counters[0] / requests),
        "l2_hit_rate": float(counters[1] / requests),
        "miss_rate": float(counters[2] / requests),
        "backend_bytes": int(counters[3]),
        "l2_write_bytes": int(counters[4]),
        "traffic_proxy_per_request": float(
            (counters[3] + 0.4 * counters[4]) / requests
        ),
    }


def take(trace, start, end):
    return tuple(array[start:end] for array in trace)


def main(args):
    warmup = 12000
    if args.real:
        source, target = [public_trace(cluster, args.records) for cluster in (17, 52)]
    else:
        source, target = (
            generate(args.seed, args.records),
            generate(args.seed + 1, args.records),
        )
    cutoff = int(0.68 * len(source[0]))
    train = take(source, 0, cutoff)
    validation = take(source, cutoff, len(source[0]))
    shape = tuple(value * 1024 for value in (96, 1024))
    x, y = make_training_rows(train, shape, warmup)
    model = HistGradientBoostingClassifier(
        max_iter=110,
        max_leaf_nodes=39,
        learning_rate=0.08,
        early_stopping=False,
        l2_regularization=1.0,
        random_state=args.seed,
    ).fit(x, y)
    boundaries, lookup = scores_lookup(model, x, args.seed)
    x_val, y_val = make_training_rows(validation, shape, warmup)
    arm = (boundaries, lookup)
    thresholds = (0.18, 0.28, 0.40, 0.54, 0.66)
    cases = {
        str(threshold): replay(
            validation, shape, warmup, "learned", arm, threshold=threshold
        )
        for threshold in thresholds
    }
    threshold = float(
        min(cases, key=lambda key: cases[key]["traffic_proxy_per_request"])
    )
    output = ROOT / (
        "real-data-projects/jones-twitter-cache-admission"
        if args.real
        else "professor-aligned/jones-two-tier-memory-traffic"
    )
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "lookup": arm, "threshold": threshold},
        output / "admission.joblib",
        compress=3,
    )
    report = {
        "configuration": vars(args),
        "source_trace_records": len(source[0]),
        "target_trace_records": len(target[0]),
        "training_eviction_examples": len(x),
        "validation_eviction_examples": len(x_val),
        "validation_ap": float(
            average_precision_score(y_val, model.predict_proba(x_val)[:, 1])
        ),
        "validation_threshold_scores": cases,
        "threshold": threshold,
        "held_out": {},
    }
    reference = feature_history(*target)
    for l1, l2 in ((64, 512), (96, 1024), (128, 2048)):
        capacities = (l1 * 1024, l2 * 1024)
        report["held_out"][f"{l1}KiB+{l2}KiB"] = {
            "lru": replay(target, capacities, warmup, "lru"),
            "second_touch": replay(
                target, capacities, warmup, "second_touch", ref=reference
            ),
            "learned": replay(
                target, capacities, warmup, "learned", arm, threshold=threshold
            ),
        }
        print(report["held_out"][f"{l1}KiB+{l2}KiB"], flush=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--records", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=63)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
