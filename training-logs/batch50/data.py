from datetime import datetime
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def pack(x, y, g, kind="classification", outputs=2, **extras):
    return dict(
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y),
        groups=np.asarray(g),
        kind=kind,
        outputs=outputs,
        **extras,
    )


def med(root, name, count, train=True):
    part = "train" if train else "test"
    with np.load(root / f"medmnist/{name}.npz") as f:
        x = f[part + "_images"]
        y = f[part + "_labels"].ravel()
    if len(y) > count:
        rng = np.random.default_rng(781 if train else 782)
        ids = rng.choice(len(y), count, replace=False)
        x = x[ids]
        y = y[ids]
    x = x.astype(np.float32) / 255.0
    return (x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)), y


def phantom(n, classes, rgb=False):
    rng = np.random.default_rng(909)
    y = np.arange(n) % classes
    rng.shuffle(y)
    u, v = np.mgrid[-1:1:28j, -1:1:28j]
    x = []
    for i in y:
        a, b = rng.normal(0, 0.18, 2)
        blob = np.exp(
            -((u - a) ** 2 + ((v - b) / (1 + 0.12 * i)) ** 2) / (0.13 + 0.012 * i)
        )
        texture = 0.12 * np.cos((2 + i * 0.5) * np.pi * u + v * (2 + 0.3 * i))
        f = 0.38 + (0.23 + 0.025 * i) * blob + texture + rng.normal(0, 0.12, u.shape)
        x.append(
            np.stack([f, f + 0.06 * i * blob, f - 0.04 * i * texture])
            if rgb
            else f[None]
        )
    return np.clip(np.asarray(x), 0, 1), y


def glucose(root, real):
    rng = np.random.default_rng(54)
    people = {}
    if real:
        for p in sorted((root / "diabetes-glucose/Diabetes-Data").glob("data-*")):
            readings = {}
            for line in p.read_text(errors="ignore").splitlines():
                fields = line.split()
                if len(fields) != 4 or fields[2] not in (
                    "48",
                    "57",
                    "58",
                    "59",
                    "60",
                    "61",
                    "62",
                    "63",
                    "64",
                ):
                    continue
                try:
                    stamp = datetime.strptime(
                        fields[0] + " " + fields[1], "%m-%d-%Y %H:%M"
                    )
                    z = float(fields[3])
                except ValueError:
                    continue
                if 35 <= z <= 550:
                    readings[stamp] = z
            if readings:
                days = sorted(readings)
                t = np.asarray(
                    [
                        (stamp - days[0].replace(hour=0, minute=0)).total_seconds()
                        / 3600
                        for stamp in days
                    ]
                )
                people[p.name] = t, np.asarray([readings[stamp] for stamp in days])
    else:
        for p in range(90):
            t = np.arange(0, 24 * 28, 3.0)
            h = t % 24
            z = (
                rng.normal(125, 22)
                + 30 * np.cos((h - 14) / 24 * 2 * np.pi)
                + rng.normal(0, 24, len(t))
            )
            people[p] = t, z
    x = []
    y = []
    g = []
    for p, (t, z) in people.items():
        for j in range(5, len(t)):
            h = t[j] % 24
            if h < 19 or h > 23.99:
                continue
            day = np.floor(t[j] / 24) * 24
            later = (t >= day + 24) & (t < day + 30)
            if not later.any():
                continue
            times = t[j - 5 : j + 1]
            v = z[j - 5 : j + 1]
            x.append(
                [
                    v / 100,
                    np.log1p(t[j] - times),
                    np.sin(times * np.pi / 12),
                    np.cos(times * np.pi / 12),
                ]
            )
            y.append(int(np.min(z[later]) < 70))
            g.append(p)
    assert len(set(g)) >= 20 and len(set(y)) == 2, (len(g), len(set(g)), sum(y))
    return pack(x, y, g)


def locate(root, real):
    rng = np.random.default_rng(425)
    if real:
        base = root / "kvasir-seg/extracted/Kvasir-SEG/images"
        files = sorted(base.glob("*.jpg"))[::3]
        backgrounds = []
        for p in files:
            with Image.open(p) as img:
                backgrounds.append(
                    np.asarray(img.convert("L").resize((40, 40))) / 255.0
                )
    else:
        source, _ = phantom(420, 4)
        backgrounds = ndi.zoom(source[:, 0], (1, 40 / 28, 40 / 28), order=1)
    yy, xx = np.mgrid[:40, :40]
    x = []
    y = []
    g = []
    gx = (xx - 19.5) / 20
    gy = (yy - 19.5) / 20
    for group, base in enumerate(backgrounds):
        for _ in range(3):
            u, v = rng.uniform(7, 33, 2)
            marker = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / rng.uniform(2.7, 6.0))
            image = np.clip(
                0.58 * base + 0.42 * marker + rng.normal(0, 0.025, base.shape), 0, 1
            )
            x.append(np.stack((image, gx, gy)))
            y.append([(u - 19.5) / 20, (v - 19.5) / 20])
            g.append(group)
    return pack(x, y, g, "coordinates", 2)


def reconstruct(root, real):
    src = med(root, "organamnist", 340)[0] if real else phantom(375, 6)[0]
    field = []
    angles = np.linspace(0, 180, 9, endpoint=False)
    for im in src[:, 0]:
        padded = np.pad(im, 6)
        acc = np.zeros_like(padded)
        for angle in angles:
            rotated = ndi.rotate(padded, angle, reshape=False, order=1)
            profile = rotated.mean(0)
            filt = np.fft.irfft(
                np.fft.rfft(profile) * np.fft.rfftfreq(len(profile)), n=len(profile)
            )
            back = ndi.rotate(
                np.broadcast_to(filt, padded.shape), -angle, reshape=False, order=1
            )
            acc += back
        crop = acc[6:-6, 6:-6]
        field.append(
            np.clip(crop / (np.quantile(np.abs(crop), 0.99) + 1e-5) * 0.5 + 0.35, 0, 1)
        )
    x = np.asarray(field, dtype=np.float32)[:, None]
    return pack(x, src, np.arange(len(src)), "image", 1, baseline=x)


def image_classes(root, idx, real):
    names = {3: "breastmnist", 6: "pathmnist", 9: "pneumoniamnist_64"}
    classes = 9 if idx == 6 else 2
    if real:
        a, y = med(root, names[idx], 1900, True)
        b, z = med(root, names[idx], 520, False)
        if idx == 9:
            a = ndi.zoom(a, (1, 1, 0.5, 0.5), order=1)
            b = ndi.zoom(b, (1, 1, 0.5, 0.5), order=1)
        x = np.concatenate((a, b))
        targets = np.r_[y, z]
        te = np.arange(len(x)) >= len(a)
    else:
        x, targets = phantom(1350, classes, idx == 6)
        te = None
    opts = {}
    if idx == 3:
        rng = np.random.default_rng(193)
        opts["stress"] = np.clip(
            0.62 * ndi.gaussian_filter(x, (0, 0, 0.85, 0.85))
            + 0.09
            + rng.normal(0, 0.045, x.shape),
            0,
            1,
        )
    return pack(x, targets, np.arange(len(x)), outputs=classes, test=te, **opts)


def wearable(root, real):
    rng = np.random.default_rng(55)
    x = []
    y = []
    g = []
    if real:
        for p in sorted((root / "sleep-accel/labels").glob("*sleep.txt")):
            person = p.name.split("_")[0]
            hpath = root / "sleep-accel/heart_rate" / (person + "_heartrate.txt")
            if not hpath.exists():
                continue
            hr = np.loadtxt(hpath, delimiter=",")
            hr = hr[np.argsort(hr[:, 0])]
            rows = np.loadtxt(p)
            for j in range(9, len(rows) - 1):
                times = rows[j - 9 : j + 1, 0]
                if rows[j, 1] < 0 or rows[j + 1, 1] < 0 or np.any(np.diff(times) != 30):
                    continue
                bounds = np.searchsorted(hr[:, 0], np.r_[times, times[-1] + 30])
                feats = []
                for lo, hi in zip(bounds, bounds[1:]):
                    v = hr[lo:hi, 1]
                    feats.append(
                        (v.mean() / 100, v.std() / 25, 1.0)
                        if len(v)
                        else (0.0, 0.0, 0.0)
                    )
                if sum(a[2] for a in feats) < 5:
                    continue
                x.append(np.asarray(feats).T)
                y.append(int((rows[j, 1] == 0) != (rows[j + 1, 1] == 0)))
                g.append(person)
    else:
        for person in range(45):
            state = 0
            a = []
            for j in range(96):
                p = 0.10 + 0.07 * np.sin(j * 0.25)
                next_state = 1 - state if rng.random() < p else state
                a.append(
                    (
                        0.7 + 0.12 * state + rng.normal(0, 0.065),
                        0.12 + 0.06 * state + rng.random() * 0.12,
                        1.0,
                    )
                )
                state = next_state
            a = np.array(a)
            for j in range(9, len(a) - 1):
                x.append(a[j - 9 : j + 1].T)
                y.append(int((a[j + 1, 0] > 0.8) != (a[j, 0] > 0.8)))
                g.append(person)
    assert len(set(g)) >= 15, (len(g), len(set(g)))
    return pack(x, y, g)


def har(root, idx, real):
    rng = np.random.default_rng(255 + (idx == 7))
    if real:
        with ZipFile(root / "har.zip") as outer:
            nested = next(
                (n for n in outer.namelist() if n.lower().endswith(".zip")), None
            )
            source = ZipFile(BytesIO(outer.read(nested))) if nested else outer
            names = source.namelist()

            def fetch(suffix, dtype=float):
                name = next(n for n in names if n.endswith(suffix))
                return np.loadtxt(BytesIO(source.read(name)), dtype=dtype)

            instances = []
            labels = []
            groups = []
            te = []
            for part in ("train", "test"):
                y = fetch(f"/y_{part}.txt", int) - 1
                subject = fetch(f"/subject_{part}.txt", int)
                order = rng.choice(
                    len(y),
                    min(len(y), 2200 if part == "train" else 1000),
                    replace=False,
                )
                keys = [
                    f"body_{mod}_{axis}_{part}.txt"
                    for mod in ("acc", "gyro")
                    for axis in "xyz"
                ]
                x = np.stack([fetch(k)[order, ::2] for k in keys], axis=1)
                instances.append(x)
                labels.append(y[order])
                groups.append(subject[order])
                te.extend([part == "test"] * len(order))
            if nested:
                source.close()
        x, y, g, te = (
            np.concatenate(instances),
            np.concatenate(labels),
            np.concatenate(groups),
            np.asarray(te),
        )
    else:
        g = np.repeat(np.arange(48), 34)
        y = rng.integers(6, size=len(g))
        t = np.linspace(0, 1, 64)
        phase = rng.uniform(0, 6.28, (len(y), 6, 1))
        wave = np.sin(phase + 2 * np.pi * (1 + y[:, None, None] / 2) * t[None, None])
        x = wave * (0.15 + 0.12 * y[:, None, None]) + rng.normal(0, 0.2, wave.shape)
        te = None
    if idx == 7:
        stress = x.copy()
        stress[:, 3:] = 0
        return pack(x, y, g, outputs=6, test=te, stress=stress)
    return pack(x, y, g, outputs=6, test=te)


def vessels(root, real):
    if not real:
        rng = np.random.default_rng(818)
        u, v = np.mgrid[-1:1:64j, -1:1:64j]
        images = []
        masks = []
        for _ in range(160):
            boundary = (
                np.abs(v - 0.27 * np.sin(u * rng.uniform(3, 10) + rng.uniform(0, 6)))
                < 0.065
            )
            img = (
                0.55
                - 0.27 * boundary
                + 0.12 * np.sin(v * rng.uniform(3, 14))
                + rng.normal(0, 0.095, u.shape)
            )
            images.append(np.clip(img, 0, 1))
            masks.append(boundary)
        return pack(
            np.asarray(images)[:, None],
            np.asarray(masks)[:, None],
            np.arange(len(images)),
            "mask",
            1,
        )
    x = []
    y = []
    te = []
    for split, ids in [("train", range(21, 41)), ("val", range(1, 21))]:
        for idx in ids:
            base = root / "drive" / split
            with Image.open(base / "input" / f"{idx:02}.tif") as p:
                x.append(
                    np.asarray(p.convert("RGB").getchannel("G").resize((96, 96)))
                    / 255.0
                )
            with Image.open(
                base / "label" / f"{idx:02}{'_manual1' if split == 'val' else ''}.png"
            ) as p:
                mask = np.asarray(
                    p.convert("L").resize((96, 96), Image.Resampling.NEAREST)
                )
                y.append(mask > 0 if mask.max() == 1 else mask > 127)
            te.append(split == "val")
    return pack(
        np.asarray(x)[:, None],
        np.asarray(y)[:, None],
        np.arange(len(x)),
        "mask",
        1,
        test=np.asarray(te),
    )


def load_data(root, idx, real):
    if idx == 0:
        return glucose(root, real)
    if idx == 1:
        return locate(root, real)
    if idx == 2:
        return reconstruct(root, real)
    if idx in (3, 6, 9):
        return image_classes(root, idx, real)
    if idx == 4:
        return wearable(root, real)
    if idx in (5, 7):
        return har(root, idx, real)
    return vessels(root, real)
