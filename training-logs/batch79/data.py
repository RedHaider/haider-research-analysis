import csv
import hashlib
import re
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch50.data import med, pack, phantom, wearable
from batch60.data import diabetes, icu
from batch60.data import load as load60

sys.path.insert(0, str(Path(__file__).resolve().parent))
SEED = 1741


def split_images(root, name, real, count, classes, rgb=False, native=32):
    if real:
        a, y = med(root, name, count, True)
        b, z = med(root, name, 460, False)
        x = np.concatenate((a, b))
        y = np.r_[y, z]
        test = np.arange(len(x)) >= len(a)
    else:
        x, y = phantom(count, classes, rgb)
        test = None
    if x.shape[-1] != native:
        ratio = native / x.shape[-1]
        x = ndi.zoom(x, (1, 1, ratio, ratio), order=1)
    return x, y, test


def readmission(root, mode, real):
    d = diabetes(root, 2, real)
    x = d["x"]
    rng = np.random.default_rng(SEED)
    # Three encounter-level views with different feature treatment.
    if mode == "diabetes":
        absent = x.copy()
        absent[:, 2:5] = 0
        d["stress"] = absent
    elif mode == "fair":
        # Hold measurements the same in every arm. The wider model uses age-group weights.
        d["age"] = (x[:, 0]).astype(int)
    else:
        # Preserve the stream and compare models with a deliberately small training cap.
        order = rng.permutation(len(x))[: min(len(x), 6200)]
        d = {
            k: v[order] if isinstance(v, np.ndarray) and len(v) == len(x) else v
            for k, v in d.items()
        }
    return d


def heart(root, real):
    rng = np.random.default_rng(SEED)
    if not real:
        x = rng.normal(size=(1100, 13))
        site = np.arange(1100) % 4
        x[:, 0] += 0.5 * site
        y = rng.binomial(1, expit(-1 + x[:, 0] + 0.75 * x[:, 3] - 0.2 * site * x[:, 5]))
    else:
        arrays = []
        sites = []
        for i, p in enumerate(sorted((root / "heart-sites").glob("processed.*.data"))):
            v = np.genfromtxt(
                p, delimiter=",", missing_values="?", filling_values=np.nan
            )
            arrays.append(v)
            sites.extend([i] * len(v))
        data = np.vstack(arrays)
        site = np.asarray(sites)
        x = data[:, :13]
        y = (data[:, 13] > 0).astype(int)
    missing = ~np.isfinite(x)
    return pack(np.c_[np.nan_to_num(x), missing], y, np.arange(len(y)), test=site == 1)


def text(root, real):
    rng = np.random.default_rng(SEED)
    samples = []
    targets = []
    test = []
    if real:
        src = (
            root.parent
            / "real-data-projects/roco-radiology-retrieval/data/sample_manifest.csv"
        )
        tokens = re.compile(
            r"\b(no|without|absent|neither|negative|free of|absence of)\b", re.I
        )
        with src.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                examples = row["caption"]
                samples.append(tokens.sub(" ", examples))
                targets.append(int(tokens.search(examples) is not None))
                test.append(row["split"] == "test")
        # Negation is weakly labeled by a cue lexicon; cue terms are removed from inputs.
        if len(set(np.asarray(targets)[np.asarray(test)])) < 2:
            raise ValueError("No positive captions in official test subset")
    else:
        findings = [
            "edema",
            "mass",
            "pleural fluid",
            "infiltrate",
            "calcification",
            "opacity",
        ]
        for _ in range(1100):
            target = int(rng.random() < 0.32)
            f = rng.choice(findings)
            context = (
                rng.choice(["unchanged", "clear", "resolved", "follow-up", "normal"])
                if target
                else rng.choice(
                    ["progressive", "localized", "diffuse", "follow-up", "stable"]
                )
            )
            samples.append(
                f"{context} {f} views of {rng.choice(['chest', 'abdomen', 'torso'])}"
            )
            targets.append(target)
            test.append(False)
    vec = HashingVectorizer(
        n_features=1024,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )
    x = vec.transform(samples).toarray()
    g = np.arange(len(samples))
    if not real:
        g = np.array(samples, dtype=str)
    return pack(x, targets, g, test=np.asarray(test) if real else None)


def colored(root, mode, real):
    sources = {
        "image": ("pathmnist", 9, True, 36),
        "spatial": ("pathmnist", 9, True, 32),
        "breast": ("breastmnist", 2, False, 32),
        "retina": ("retinamnist", 5, True, 32),
        "quality": ("pneumoniamnist_64", 3, False, 64),
    }
    src, classes, rgb, native = sources[mode]
    x, y, test = split_images(
        root, src, real, 1400 if mode != "quality" else 900, classes, rgb, native
    )
    rng = np.random.default_rng(SEED)
    groups = np.arange(len(x))
    kw = {}
    if mode == "image":
        gains = rng.uniform(0.55, 1.5, (len(x), x.shape[1], 1, 1))
        kw["stress"] = np.clip(x * gains + 0.055, 0, 1)
    if mode == "spatial":
        row, col = np.mgrid[-1:1:32j, -1:1:32j]
        coords = np.broadcast_to(np.stack((row, col)), (len(x), 2, 32, 32))
        x = np.concatenate((x, coords), axis=1)
        shuffled = x.copy()
        for j in range(0, 32, 16):
            shuffled[:, :3, j : j + 16] = x[:, :3, 16 - j : 32 - j]
        kw["stress"] = shuffled
    if mode == "retina":
        y = (y >= 2).astype(int)
        classes = 2
    if mode == "breast":
        kw["stress"] = np.clip(
            ndi.gaussian_filter(x, (0, 0, 0.9, 0.9)) * 0.65
            + 0.08
            + rng.normal(0, 0.05, x.shape),
            0,
            1,
        )
    if mode == "quality":
        y = rng.integers(3, size=len(x))
        for k in range(3):
            use = y == k
            sigma = [0, 0.8, 1.8][k]
            x[use] = np.clip(
                ndi.gaussian_filter(x[use], (0, 0, sigma, sigma))
                + rng.normal(0, [0.01, 0.04, 0.075][k], x[use].shape),
                0,
                1,
            )
    return pack(x, y, groups, outputs=classes, test=test, **kw)


def mri(root, real):
    if not real:
        x, y = phantom(370, 2)
        return ndi.zoom(x, (1, 1, 64 / 28, 64 / 28), order=1), y, np.arange(len(y))
    seen = set()
    x = []
    y = []
    g = []
    with ZipFile(root / "brain-mri-2d.zip") as z:
        paths = sorted(
            p
            for p in z.namelist()
            if Path(p).parent.name.lower() in ("yes", "no")
            and p.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        for path in paths:
            content = z.read(path)
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            with Image.open(BytesIO(content)) as im:
                x.append(
                    np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32) / 255
                )
            y.append(int(Path(path).parent.name.lower() == "yes"))
            g.append(digest)
    assert len(x) >= 200
    return np.asarray(x)[:, None], np.asarray(y), np.asarray(g)


def mri_study(root, mode, real):
    x, y, g = mri(root, real)
    if mode == "mri_class":
        return pack(x, y, g)
    rng = np.random.default_rng(SEED)
    if mode == "mri_freq":
        keep = np.zeros(64, dtype=bool)
        keep[[0, 32]] = True
        offsets = np.r_[1, 2, 3, rng.choice(np.arange(4, 32), 4, replace=False)]
        keep[32 + offsets] = True
        keep[32 - offsets] = True
        data = (
            np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))
            * keep[None, None, :, None]
        )
        raw = np.fft.ifft2(
            np.fft.ifftshift(data, axes=(-2, -1)), axes=(-2, -1)
        ).real.astype(np.float32)
        mask = np.broadcast_to(keep[None, None, :, None], x.shape).astype(np.float32)
        return pack(
            np.concatenate((raw, mask), 1), x, g, "image", 1, baseline=raw, kspace=keep
        )
    noisy = np.sqrt(
        (x + rng.normal(0, 0.11, x.shape)) ** 2 + rng.normal(0, 0.11, x.shape) ** 2
    )
    raw = np.clip(noisy, 0, 1)
    return pack(raw, x, g, "image", 1, baseline=raw)


def flow(root, real):
    if not real:
        images, _ = phantom(500, 5)
        source = ndi.zoom(images[:, 0], (1, 40 / 28, 40 / 28), order=1)
    else:
        files = sorted((root / "kvasir-seg/extracted/Kvasir-SEG/images").glob("*.jpg"))[
            ::2
        ]
        source = []
        for p in files:
            with Image.open(p) as im:
                source.append(np.asarray(im.convert("L").resize((40, 40))) / 255)
    rng = np.random.default_rng(SEED)
    u, v = np.mgrid[:40, :40]
    x = []
    y = []
    groups = []
    for i, img in enumerate(source):
        for _ in range(2):
            dx = rng.uniform(-3, 3) + rng.uniform(0.7, 2.8) * np.sin(
                u / 8 + rng.uniform(0, 6)
            )
            dy = rng.uniform(-3, 3) + rng.uniform(0.7, 2.8) * np.cos(
                v / 8 + rng.uniform(0, 6)
            )
            warped = ndi.map_coordinates(img, [u + dy, v + dx], order=1, mode="reflect")
            x.append(np.stack((img, warped)))
            y.append(np.stack((dy, dx)) / 8)
            groups.append(i)
    return pack(x, y, groups, "flow", 2, baseline=np.zeros_like(y))


def sleep(root, real):
    if not real:
        d = wearable(root, False)
        d["y"] = (d["x"][:, 0, -1] > 0.8).astype(int)
        return d
    xs = []
    ys = []
    groups = []
    for path in sorted((root / "sleep-accel/labels").glob("*sleep.txt")):
        subject = path.name.split("_")[0]
        hp = root / "sleep-accel/heart_rate" / (subject + "_heartrate.txt")
        if not hp.exists():
            continue
        hr = np.loadtxt(hp, delimiter=",")
        hr = hr[np.argsort(hr[:, 0])]
        lab = np.loadtxt(path)
        local = []
        for j in range(7, len(lab), 3):
            times = lab[j - 7 : j + 1, 0]
            if lab[j, 1] < 0 or np.any(np.diff(times) != 30):
                continue
            indices = np.searchsorted(hr[:, 0], np.r_[times, times[-1] + 30])
            vals = []
            for a, b in zip(indices, indices[1:]):
                z = hr[a:b, 1]
                vals.append((z.mean() / 100, z.std() / 25, 1) if len(z) else (0, 0, 0))
            if sum(v[2] for v in vals) >= 4:
                local.append((np.asarray(vals).T, int(lab[j, 1] != 0)))
        for v, y in local[:180]:
            xs.append(v)
            ys.append(y)
            groups.append(subject)
    return pack(xs, ys, groups)


def horizon(root, real):
    rng = np.random.default_rng(SEED)
    x = []
    y = []
    if real:
        with (root / "lung-survival.csv").open() as f:
            records = csv.DictReader(f)
            for rec in records:
                t = float(rec["time"])
                event = int(rec["status"]) == 1
                if t < 180 and not event:
                    continue
                vals = [
                    float(rec[k]) if rec[k] else np.nan
                    for k in (
                        "age",
                        "sex",
                        "ph.ecog",
                        "ph.karno",
                        "pat.karno",
                        "meal.cal",
                        "wt.loss",
                    )
                ]
                x.append(np.r_[np.nan_to_num(vals), ~np.isfinite(vals)])
                y.append(int(t <= 180 and event))
    else:
        x = rng.normal(0, 1, (950, 14))
        y = rng.binomial(1, expit(-0.6 + x[:, 0] + 0.8 * x[:, 1] * x[:, 2]))
    return pack(x, y, np.arange(len(y)))


def eeg(root, real):
    rng = np.random.default_rng(SEED)
    if not real:
        t = np.arange(128) / 64
        y = rng.integers(2, size=1200)
        x = rng.normal(0, 0.8, (len(y), 8, len(t)))
        x += np.sin(2 * np.pi * (9 + 6 * y[:, None, None]) * t).reshape(
            len(y), 1, len(t)
        )
        g = np.arange(len(y)) // 20
    else:
        import edfio
        from scipy.signal import butter, resample_poly, sosfiltfilt

        xs = []
        ys = []
        groups = []
        for pid in range(1, 17):
            for run in (4, 8):
                f = edfio.read_edf(root / "eegmmidb" / f"S{pid:03}R{run:02}.edf")
                signals = {s.label.strip(".").lower(): s.data for s in f.signals}
                x = np.stack(
                    [signals[k] for k in "fc3 fc1 c3 c1 fc4 fc2 c4 c2".split()]
                )
                x = sosfiltfilt(
                    butter(4, (7, 26), fs=160, btype="bandpass", output="sos"), x
                )
                for event in f.annotations:
                    if event.text not in ("T1", "T2"):
                        continue
                    j = round((event.onset + 0.5) * 160)
                    xs.append(resample_poly(x[:, j : j + 320], 2, 5, axis=1))
                    ys.append(int(event.text == "T2"))
                    groups.append(pid)
        x, y, g = np.asarray(xs), np.asarray(ys), np.asarray(groups)
    reduced = ndi.gaussian_filter1d(x, 1.4, axis=-1)
    return pack(x, y, g, stress=reduced)


def pair(root, real):
    d = load60(root, 1, real)
    rng = np.random.default_rng(SEED)
    x = d["x"]
    a, b = x[:, :512], x[:, 512:1024]
    a = a + rng.normal(0, 0.012, a.shape)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    d["stress"] = np.concatenate((a, b, np.abs(a - b), a * b), axis=1)
    return d


def load_task(root, mode, real):
    if mode in ("diabetes", "fair", "scale"):
        return readmission(root, mode, real)
    if mode == "heart":
        return heart(root, real)
    if mode == "text":
        return text(root, real)
    if mode in ("image", "spatial", "breast", "retina", "quality"):
        return colored(root, mode, real)
    if mode == "icu":
        d = icu(root, 8, real)
        # Suppress lab measurements and their observation flags together.
        d["stress"] = d["x"].copy()
        d["stress"][:, :6] = 0
        d["stress"][:, 8:14] = 0
        return d
    if mode.startswith("mri"):
        return mri_study(root, mode, real)
    if mode == "flow":
        return flow(root, real)
    if mode == "sleep":
        return sleep(root, real)
    if mode == "survival":
        return horizon(root, real)
    if mode == "eeg":
        return eeg(root, real)
    return pair(root, real)
