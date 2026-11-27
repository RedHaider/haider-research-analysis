import csv
import re
import sys
import tarfile
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch50.data import har, med, pack, phantom


def clean_images(root, index, real):
    rng = np.random.default_rng(80)
    if not real:
        src, _ = phantom(1050, 6, index == 0)
        return (
            ndi.zoom(src, (1, 1, 64 / 28, 64 / 28), order=1),
            np.arange(len(src)),
            None,
        )
    if index == 0:
        x = []
        groups = []
        with ZipFile(root / "malaria-cell-images.zip") as z:
            names = sorted(
                p
                for p in z.namelist()
                if p.lower().endswith(".png") and re.match(r"C\d+", Path(p).name)
            )
            for i in rng.choice(len(names), 1500, replace=False):
                p = names[i]
                with Image.open(BytesIO(z.read(p))) as im:
                    x.append(np.asarray(im.convert("RGB").resize((64, 64))) / 255.0)
                groups.append(re.match(r"C\d+", Path(p).name).group())
        return (
            np.asarray(x, dtype=np.float32).transpose(0, 3, 1, 2),
            np.asarray(groups),
            None,
        )
    train, _ = med(root, "pneumoniamnist_64", 1250, True)
    test, _ = med(root, "pneumoniamnist_64", 280, False)
    src = np.concatenate((train, test))
    return src, np.arange(len(src)), np.arange(len(src)) >= len(train)


def restoration(root, index, real):
    src, group, test = clean_images(root, index, real)
    src = src.astype(np.float32)
    rng = np.random.default_rng(199 + index)
    if index == 0:
        inp = ndi.gaussian_filter(src, (0, 0, 0.7, 0.7))
        inp = np.clip(inp + rng.normal(0, 0.095, src.shape), 0, 1)
    elif index == 6:
        mask = rng.random(src[:, 0:1].shape) > 0.76
        blurred = ndi.gaussian_filter(src * mask, (0, 0, 1.3, 1.3)) / np.maximum(
            ndi.gaussian_filter(mask.astype(np.float32), (0, 0, 1.3, 1.3)), 0.03
        )
        inp = np.concatenate((np.where(mask, src, blurred), mask), axis=1)
    else:
        box = ndi.uniform_filter1d(src, 9, axis=3, mode="reflect")
        inp = np.clip(box + rng.normal(0, 0.018, src.shape), 0, 1)
    return pack(
        inp,
        src,
        group,
        "image",
        src.shape[1],
        test=test,
        baseline=inp[:, : src.shape[1]],
    )


def multimodal(root, real):
    rng = np.random.default_rng(889)
    parts = []
    if real:
        with np.load(root / "roco_clip_embeddings.npz") as a:
            for k in ["train", "test"]:
                parts.append((a[k + "_image"], a[k + "_text"]))
    else:
        mapping = None
        for n in (720, 160):
            z = rng.normal(0, 1, (n, 96))
            mapping = rng.normal(0, 1, (96, 512)) if mapping is None else mapping
            shared = z @ mapping / np.sqrt(96)
            parts.append(
                (
                    shared + rng.normal(0, 1.5, shared.shape),
                    shared + rng.normal(0, 1.5, shared.shape),
                )
            )
    x = []
    y = []
    g = []
    test = []
    base = []
    for part, (a, b) in enumerate(parts):
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        for j in range(0, len(a), 10):
            stop = min(j + 10, len(a))
            n = stop - j
            if n < 2:
                continue
            scores = a[j:stop] @ b[j:stop].T
            for k in range(n):
                negatives = np.argsort(-scores[k])
                negatives = negatives[negatives != k][:3]
                for v in np.r_[k, negatives]:
                    x.append(
                        np.r_[
                            a[j + k],
                            b[j + v],
                            np.abs(a[j + k] - b[j + v]),
                            a[j + k] * b[j + v],
                        ]
                    )
                    y.append(int(v == k))
                    g.append(f"{part}:{j}")
                    test.append(part == 1)
                    base.append([-scores[k, v], scores[k, v]])
    return pack(
        x, y, g, test=np.asarray(test), baseline=np.asarray(base, dtype=np.float32)
    )


def diabetes(root, index, real):
    rng = np.random.default_rng(312 + index)
    if not real:
        x = rng.normal(size=(2600, 15))
        age = rng.integers(0, 10, len(x))
        x[:, 0] = age
        site = rng.integers(0, 4, len(x))
        p = 1 / (
            1
            + np.exp(-(-1.5 + x[:, 1] - 0.6 * x[:, 4] + 0.18 * age + 0.5 * (site == 3)))
        )
        return pack(x, rng.binomial(1, p), np.arange(len(x)) // 2, age=age)
    x = []
    y = []
    g = []
    ages = []
    with (
        ZipFile(root / "diabetes130.zip") as z,
        TextIOWrapper(z.open("diabetic_data.csv")) as data,
    ):
        for row in csv.DictReader(data):
            if rng.random() > 0.18:
                continue
            g.append(int(row["patient_nbr"]))
            age = int(row["age"].split("-")[0].lstrip("[")) // 10
            ages.append(age)
            if index == 2:
                keys = [
                    "time_in_hospital",
                    "number_inpatient",
                    "number_outpatient",
                    "number_emergency",
                    "num_lab_procedures",
                    "num_medications",
                    "num_procedures",
                    "number_diagnoses",
                    "admission_type_id",
                    "admission_source_id",
                ]
                y.append(int(row["readmitted"] == "<30"))
            else:
                # Admission-time fields and prior utilization only.
                keys = [
                    "admission_type_id",
                    "admission_source_id",
                    "number_inpatient",
                    "number_outpatient",
                    "number_emergency",
                ]
                y.append(int(float(row["time_in_hospital"]) >= 7))
            diag = (
                [
                    float(row["diag_1"].split(".")[0]) / 100
                    if row["diag_1"].split(".")[0].isdigit()
                    else -1
                ]
                if index == 2
                else []
            )
            x.append(
                [
                    age,
                    int(row["gender"] == "Male"),
                    *[float(row[k]) for k in keys],
                    *diag,
                ]
            )
    return pack(x, y, g, age=np.asarray(ages))


def icu(root, index, real):
    if not real:
        rng = np.random.default_rng(96 + index)
        n = 2250
        trend = rng.normal(size=n)
        y = (trend + rng.normal(0, 1.05, n) > 0.7).astype(int)
        times = np.linspace(0, 1, 24)
        grid = rng.normal(0, 0.55, (n, 8, 24)) + 0.35 * trend[:, None, None] * times
        seen = rng.random(grid.shape) > 0.25
        return pack(np.concatenate((grid * seen, seen), 1), y, np.arange(n))
    early_hours = 6 if index == 3 else 24
    catalog = {
        "SysABP": (120, 30),
        "NISysABP": (120, 30),
        "MAP": (80, 20),
        "HR": (80, 30),
        "RespRate": (16, 7),
        "SpO2": (95, 9),
        "Temp": (37, 2),
        "SaO2": (95, 9),
    }
    if index == 8:
        catalog = {
            "BUN": (20, 18),
            "Creatinine": (1.2, 1.5),
            "Glucose": (120, 70),
            "HCT": (35, 10),
            "K": (4, 1),
            "Na": (138, 7),
            "WBC": (9, 6),
            "Platelets": (210, 120),
        }
    fields = list(catalog)
    base = root / "icu2012"
    with (base / "Outcomes-a.txt").open() as f:
        labels = {r["RecordID"]: int(r["Length_of_stay"]) for r in csv.DictReader(f)}
    x = []
    y = []
    groups = []
    with tarfile.open(base / "set-a.tar.gz") as tar:
        for entry in tar:
            pid = Path(entry.name).stem
            if not entry.isfile() or pid not in labels:
                continue
            values = np.zeros((len(fields), 24), dtype=np.float32)
            counts = np.zeros_like(values)
            future = []
            prev = []
            age = 0.0
            sex = 0.0
            with TextIOWrapper(tar.extractfile(entry)) as f:
                for r in csv.DictReader(f):
                    name = r["Parameter"]
                    v = float(r["Value"])
                    if name == "Age":
                        age = max(0.0, v / 100)
                    if name == "Gender":
                        sex = max(0.0, v)
                    if name not in catalog or v <= 0:
                        continue
                    h, m = map(int, r["Time"].split(":"))
                    t = h + m / 60
                    if name in ("SysABP", "NISysABP") and 40 <= v <= 280:
                        if t <= 6:
                            prev.append(v)
                        elif t <= 12:
                            future.append(v)
                    if t >= early_hours:
                        continue
                    b = min(23, int(t * 24 / early_hours))
                    i = fields.index(name)
                    offset, scale = catalog[name]
                    values[i, b] += np.clip((v - offset) / scale, -5, 5)
                    counts[i, b] += 1
            if index == 3 and (not prev or not future or min(prev) < 90):
                continue
            biom = np.concatenate((values / np.maximum(counts, 1), counts > 0))
            if index == 8:
                biom = np.r_[biom, np.full((1, 24), age), np.full((1, 24), sex)]
            x.append(biom)
            groups.append(pid)
            y.append(int(min(future) < 90) if index == 3 else int(labels[pid] >= 7))
    return pack(x, y, groups)


def mask(root, real):
    if not real:
        src, y = phantom(450, 2, True)
        src = ndi.zoom(src, (1, 1, 64 / 28, 64 / 28), order=1)
        m = ndi.gaussian_filter(src[:, 0], (0, 3, 3)) > 0.60
    else:
        base = root / "kvasir-seg/extracted/Kvasir-SEG"
        images = []
        masks = []
        for path in sorted((base / "images").glob("*.jpg"))[::2]:
            with Image.open(path) as im:
                images.append(np.asarray(im.convert("RGB").resize((64, 64))) / 255)
            with Image.open(base / "masks" / path.name) as im:
                masks.append(
                    np.asarray(
                        im.convert("L").resize((64, 64), Image.Resampling.NEAREST)
                    )
                    > 127
                )
        src = np.asarray(images).transpose(0, 3, 1, 2)
        m = np.asarray(masks)
    return pack(src, m[:, None], np.arange(len(src)), "mask", 1)


def load(root, index, real):
    if index in (0, 6, 7):
        return restoration(root, index, real)
    if index == 1:
        return multimodal(root, real)
    if index in (2, 4):
        return diabetes(root, index, real)
    if index in (3, 8):
        return icu(root, index, real)
    if index == 5:
        return mask(root, real)
    data = har(root, 5, real)
    rng = np.random.default_rng(99)
    x = data["x"]
    drift = rng.uniform(0.65, 1.45, (len(x), x.shape[1], 1))
    data["stress"] = (
        x * drift
        + 0.17 * np.sin(np.linspace(0, 2 * np.pi, x.shape[-1]))
        + 0.04 * rng.normal(size=x.shape)
    )
    return data
