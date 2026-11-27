import csv
import sys
import tarfile
from collections import defaultdict
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch50.data import har, med, pack, phantom, vessels
from batch60.data import diabetes, icu, mask, multimodal
from batch79.data import eeg, heart, mri


def clinical(root, index, real):
    rng = np.random.default_rng(610 + index)
    if index == 0:
        d = icu(root, 8, real)
        x = d["x"]
        stress = x.copy()
        stress[:, :4] = 0
        stress[:, 8:12] = 0
        d["stress"] = stress
        return d
    if index == 1:
        d = heart(root, real)
        x = d["x"]
        altered = x.copy()
        altered[:, 3:6] = 0
        d["stress"] = altered
        return d
    d = diabetes(root, 2, real)
    x = d["x"]
    ids = rng.permutation(len(x))[:4200]
    d = {
        k: v[ids] if isinstance(v, np.ndarray) and len(v) == len(x) else v
        for k, v in d.items()
    }
    if index == 3:
        x = d["x"]
        observed = rng.random(x.shape) > 0.13
        observed[:, :2] = True
        d["x"] = np.c_[x * observed, observed].astype(np.float32)
    return d


def breast(root, index, real):
    if not real:
        x, y = phantom(950, 2)
        x = ndi.zoom(x, (1, 1, 64 / 28, 64 / 28), order=1)
        test = None
    else:
        a, y = med(root, "breastmnist_64", 700, True)
        b, z = med(root, "breastmnist_64", 200, False)
        x = np.concatenate((a, b))
        y = np.r_[y, z]
        test = np.arange(len(x)) >= len(a)
    if index == 2:
        z = ndi.gaussian_filter(x, (0, 0, 1.5, 1.5))
        texture = np.abs(x - z) * 6
        x = np.concatenate((x, texture), 1)
    else:
        rng = np.random.default_rng(911)
        noise = np.clip(x * (1 + 0.23 * rng.normal(size=x.shape)), 0, 1)
        return pack(x, y, np.arange(len(x)), test=test, stress=noise)
    return pack(x, y, np.arange(len(x)), test=test)


def glucose(root, real):
    rng = np.random.default_rng(876)
    people = {}
    if real:
        for file in sorted((root / "diabetes-glucose/Diabetes-Data").glob("data-*")):
            vals = {}
            for line in file.read_text(errors="replace").splitlines():
                rs = line.split()
                if len(rs) != 4 or rs[2] not in (
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
                    when = datetime.strptime(" ".join(rs[:2]), "%m-%d-%Y %H:%M")
                    g = float(rs[3])
                except ValueError:
                    continue
                if 35 <= g <= 600:
                    vals[when] = g
            t = sorted(vals)
            if t:
                people[file.name] = (
                    np.asarray([(z - t[0]).total_seconds() / 3600 for z in t]),
                    np.asarray([vals[z] for z in t]),
                )
    else:
        for i in range(70):
            t = np.cumsum(rng.uniform(1, 5, 100))
            value = (
                rng.normal(155, 28)
                + 38 * np.sin(t / 24 * 2 * np.pi)
                + rng.normal(0, 23, len(t))
            )
            people[i] = (t, value)
    x = []
    y = []
    groups = []
    for idx, (time, values) in people.items():
        choices = np.arange(7, len(time) - 1)
        choices = choices[
            (time[choices + 1] - time[choices] >= 0.5)
            & (time[choices + 1] - time[choices] <= 8)
        ]
        if len(choices) > 65:
            choices = np.sort(rng.choice(choices, 65, replace=False))
        for j in choices:
            z = values[j - 7 : j + 1]
            t = time[j - 7 : j + 1]
            x.append(
                np.stack(
                    (
                        z / 100,
                        np.log1p(t[-1] - t),
                        np.sin(t / 24 * 2 * np.pi),
                        np.cos(t / 24 * 2 * np.pi),
                    )
                )
            )
            y.append(int(values[j + 1] >= 180))
            groups.append(idx)
    return pack(x, y, groups)


def registration(root, index, real):
    rng = np.random.default_rng(923 + index)
    if index == 5:
        x, _, _ = mri(root, real)
        source = x[:, 0]
    elif real:
        folder = root / "kvasir-seg/extracted/Kvasir-SEG/images"
        source = []
        for src in sorted(folder.glob("*.jpg"))[::4]:
            with Image.open(src) as im:
                source.append(
                    np.asarray(im.convert("L").resize((64, 64)), np.float32) / 255
                )
    else:
        x, _ = phantom(320, 6)
        source = ndi.zoom(x[:, 0], (1, 64 / 28, 64 / 28), order=1)
    x = []
    y = []
    group = []
    coords = np.mgrid[:64, :64].astype(np.float32)
    points = np.array([[17, 17], [17, 47], [47, 17], [47, 47]])
    yy, xx = coords / 32 - 1
    for i, img in enumerate(source):
        for _ in range(3):
            dx, dy = rng.uniform(-4, 4, 2)
            angle = rng.uniform(-0.16, 0.16)
            t = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            offset = (np.eye(2) - t) @ np.array([31.5, 31.5]) + [dy, dx]
            warped = ndi.affine_transform(img, t, offset, order=1, mode="reflect")
            if index == 5:
                target = np.array([dy / 8, dx / 8, angle / 0.2])
            else:
                target = ((points @ t.T + offset - points) / 10).ravel()
            x.append(np.stack((img, warped, yy, xx)))
            y.append(target)
            group.append(i)
    return pack(x, y, group, "parameter", len(y[0]))


def morphometry(root, real):
    rng = np.random.default_rng(953)
    by_person = defaultdict(dict)
    if real:
        from openpyxl import load_workbook

        w = load_workbook(root / "oasis2.xlsx", read_only=True, data_only=True)
        values = w.active.values
        fields = next(values)
        for row in values:
            d = dict(zip(fields, row))
            if d["Subject ID"] and d["MR Delay"] is not None and d["nWBV"] is not None:
                by_person[d["Subject ID"]][float(d["MR Delay"]) / 365.25] = float(
                    d["nWBV"]
                )
        w.close()
    else:
        for pid in range(340):
            times = np.r_[0, np.cumsum(rng.uniform(0.7, 2.0, 3))]
            ys = (
                rng.normal(0.74, 0.042)
                - times * rng.uniform(0.005, 0.03)
                + rng.normal(0, 0.008, 4)
            )
            by_person[pid] = dict(zip(times, ys))
    x = []
    y = []
    g = []
    for person, visits in by_person.items():
        if len(visits) < 3:
            continue
        t, z = np.asarray(sorted(visits.items())).T
        latest = z[-2]
        history = [
            latest,
            z[-3],
            (latest - z[-3]) / max(0.2, t[-2] - t[-3]),
            t[-2] - t[0],
            len(z) - 1,
        ]
        x.append(history)
        y.append([(z[-1] - 0.7) * 10])
        g.append(person)
    return pack(x, y, g, "volume", 1)


def activity(root, index, real):
    d = har(root, 5, real)
    rng = np.random.default_rng(901 + index)
    x = d["x"]
    if index == 7:
        # Emulate burst packet gaps without missing all channel data.
        altered = x.copy()
        for i, start in enumerate(rng.integers(5, 40, len(x))):
            altered[i, :, start : start + 16] = 0
        d["stress"] = altered
        d["y"] = (d["y"] >= 3).astype(int)
        d["outputs"] = 2
        return d
    return d


def edge(root, real):
    if real:
        a, y = med(root, "retinamnist", 1050, True)
        b, z = med(root, "retinamnist", 400, False)
        x = np.concatenate((a, b))
        y = np.r_[y, z]
        test = np.arange(len(x)) >= len(a)
    else:
        x, y = phantom(900, 5, True)
        test = None
    x = np.pad(x, ((0, 0), (0, 0), (0, 4), (0, 4)), mode="edge")
    low = ndi.zoom(ndi.zoom(x, (1, 1, 0.5, 0.5), order=1), (1, 1, 2, 2), order=0)
    return pack(x, (y >= 2).astype(int), np.arange(len(x)), test=test, low=low)


def biosignal(root, real):
    d = eeg(root, real)
    rng = np.random.default_rng(913)
    x = d["x"]
    a = x.copy()
    for i, t in enumerate(rng.integers(5, 100, len(x))):
        s = x[i].std()
        a[i, :, t : t + 18] += s * rng.normal(0, 1.6, (x.shape[1], 18))
    d["stress"] = a
    return d


def mri_restore(root, real):
    x, _, g = mri(root, real)
    keep = np.zeros(64, dtype=bool)
    keep[::4] = True
    keep[30:35] = True
    keep[[29, 35]] = True
    freq = np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))
    raw = np.fft.ifft2(
        np.fft.ifftshift(freq * keep[None, None, :, None], axes=(-2, -1)), axes=(-2, -1)
    ).real
    mask = np.broadcast_to(keep[None, None, :, None], x.shape)
    return pack(
        np.concatenate((raw, mask), 1), x, g, "image", 1, baseline=raw, kspace=keep
    )


def segment(root, index, real):
    if index == 13:
        d = mask(root, real)
        x = d["x"]
        rng = np.random.default_rng(751)
        d["stress"] = np.clip(
            ndi.gaussian_filter(x, (0, 0, 1.0, 1.0)) * 0.72
            + 0.07
            + rng.normal(0, 0.045, x.shape),
            0,
            1,
        )
        return d
    d = vessels(root, real)
    x = d["x"]
    y = d["y"]
    if x.shape[-1] != 64:
        v = 64 / x.shape[-1]
        d["x"] = ndi.zoom(x, (1, 1, v, v), order=1)
        d["y"] = ndi.zoom(y, (1, 1, v, v), order=0)
    if index == 19:
        field = []
        for m in d["y"][:, 0].astype(bool):
            dist = ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)
            field.append(np.clip(dist / 8, -1, 1))
        d["y"] = np.asarray(field, dtype=np.float32)[:, None]
        d["kind"] = "signed"
    else:
        z = ndi.gaussian_filter(d["x"], (0, 0, 1.3, 1.3))
        d["x"] = np.concatenate((d["x"], (d["x"] - z) * 5), axis=1)
    return d


def caption_pairs(root, real):
    d = multimodal(root, real)
    rng = np.random.default_rng(921)
    x = d["x"]
    a, b = x[:, :512], x[:, 512:1024]
    use = rng.random(b.shape) < 0.25
    t = b.copy()
    t[use] = 0
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
    d["stress"] = np.concatenate((a, t, np.abs(a - t), a * t), 1)
    return d


def heart_rate(root, real):
    rng = np.random.default_rng(444)
    if not real:
        x = rng.normal(0, 1, (1800, 8, 24))
        tr = x[:, 0, -3:].mean(1)
        y = rng.binomial(1, expit(-1.5 + 2 * tr))
        return pack(x, y, np.arange(len(y)))
    limits = {"HR": (82, 28), "MAP": (80, 20), "RespRate": (18, 8), "SpO2": (96, 8)}
    x = []
    y = []
    groups = []
    with tarfile.open(root / "icu2012/set-a.tar.gz") as tar:
        for member in tar:
            if not member.isfile() or not Path(member.name).stem.isdigit():
                continue
            values = np.zeros((4, 24), np.float32)
            count = np.zeros_like(values)
            late = []
            prior = []
            with TextIOWrapper(tar.extractfile(member)) as f:
                for r in csv.DictReader(f):
                    if r["Parameter"] not in limits:
                        continue
                    h, m = map(int, r["Time"].split(":"))
                    t = h + m / 60
                    u = float(r["Value"])
                    if u <= 0:
                        continue
                    name = r["Parameter"]
                    if name == "HR" and 25 <= u <= 225:
                        if 6 <= t < 12:
                            late.append(u)
                        elif t < 6:
                            prior.append(u)
                    if t >= 6:
                        continue
                    i = list(limits).index(name)
                    a, b = limits[name]
                    j = int(t * 4)
                    values[i, j] += np.clip((u - a) / b, -6, 6)
                    count[i, j] += 1
            if not late or not prior or max(prior) >= 115:
                continue
            x.append(np.concatenate((values / np.maximum(1, count), count > 0)))
            y.append(int(max(late) >= 115))
            groups.append(member.name)
    return pack(x, y, groups)


def photon(root, real):
    if real:
        src, _ = med(root, "organamnist", 440)
    else:
        src, _ = phantom(440, 5)
    src = ndi.zoom(src, (1, 1, 32 / 28, 32 / 28), order=1)
    rng = np.random.default_rng(782)
    out = []
    for img in src[:, 0]:
        padded = np.pad(img, 5)
        acc = np.zeros_like(padded)
        for angle in np.arange(0, 180, 15):
            rays = ndi.rotate(padded, angle, reshape=False, order=1).mean(0)
            power = 45 * np.exp(-2.2 * rays)
            measured = rng.poisson(power)
            atten = -np.log(np.maximum(measured, 1) / 45) / 2.2
            filt = np.fft.irfft(
                np.fft.rfft(atten) * np.fft.rfftfreq(len(atten)), n=len(atten)
            )
            acc += ndi.rotate(
                np.broadcast_to(filt, padded.shape), -angle, reshape=False, order=1
            )
        a = acc[5:-5, 5:-5]
        out.append(
            np.clip(0.32 + 0.50 * a / (np.quantile(np.abs(a), 0.98) + 1e-6), 0, 1)
        )
    a = np.asarray(out, np.float32)[:, None]
    return pack(a, src, np.arange(len(src)), "image", 1, baseline=a)


def load_task(root, index, real):
    if index in (0, 1, 3, 8):
        return clinical(root, index, real)
    if index in (2, 10):
        return breast(root, index, real)
    if index == 4:
        return glucose(root, real)
    if index in (5, 14):
        return registration(root, index, real)
    if index == 6:
        return morphometry(root, real)
    if index == 7:
        return activity(root, index, real)
    if index == 9:
        return edge(root, real)
    if index == 11:
        return biosignal(root, real)
    if index == 12:
        return mri_restore(root, real)
    if index in (13, 15, 19):
        return segment(root, index, real)
    if index == 16:
        return caption_pairs(root, real)
    if index == 17:
        return heart_rate(root, real)
    return photon(root, real)
