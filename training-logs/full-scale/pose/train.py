"""Learn direct and progressive image codes for CAD keypoint correspondences."""

import argparse
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
SIDE = 64
BITS = 6
NODES = 32


def load_models():
    vertices, colors, keys = [], [], []
    with ZipFile(ROOT / "datasets/tudl_models.zip") as archive:
        for obj in (1, 2, 3):
            text = archive.read(f"models/obj_{obj:06}.ply").decode("ascii").splitlines()
            count = int(
                next(
                    line.split()[-1]
                    for line in text
                    if line.startswith("element vertex")
                )
            )
            begin = text.index("end_header") + 1
            points = np.loadtxt(
                text[begin : begin + count],
                usecols=(0, 1, 2, 6, 7, 8),
                dtype=np.float32,
            )
            x, rgb = points[:, :3], points[:, 3:6]
            centroid = x.mean(axis=0)
            choice = np.empty(NODES, dtype=int)
            dist = np.linalg.norm(x - centroid, axis=1)
            choice[0] = dist.argmax()
            dist = np.linalg.norm(x - x[choice[0]], axis=1)
            for index in range(1, NODES):
                choice[index] = dist.argmax()
                dist = np.minimum(dist, np.linalg.norm(x - x[choice[index]], axis=1))
            vertices.append(x[::3])
            colors.append(rgb[::3])
            keys.append(x[choice])
    return vertices, colors, np.stack(keys)


def project(points, r, t, intrinsic):
    x = points @ r.T + t
    h = x @ intrinsic.T
    return h[:, :2] / np.maximum(h[:, 2:3], 1e-5), x[:, 2]


def simulation(seed, objects, colors, keypoints):
    rng = np.random.default_rng(seed)
    K = np.array([[62, 0, 31.5], [0, 62, 31.5], [0, 0, 1]], dtype=np.float32)
    records = []
    for index in range(6000):
        obj = index % len(objects)
        points = objects[obj]
        r = Rotation.random(random_state=rng).as_matrix()
        radius = np.linalg.norm(points, axis=1).max()
        t = radius * np.array(
            [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(3.2, 4.8)]
        )
        xy, z = project(points, r, t, K)
        row = xy.astype(int).clip(0, SIDE - 1)
        background = np.clip(
            rng.normal(46, 15, (SIDE, SIDE, 3)) + rng.uniform(0, 65), 0, 180
        ).astype(np.uint8)
        order = np.argsort(z)[::-1]
        sprite = colors[obj] * rng.uniform(0.65, 1.2) + rng.uniform(0, 32)
        background[row[order, 1], row[order, 0]] = (
            sprite[order].clip(0, 255).astype(np.uint8)
        )
        image = cv2.medianBlur(background, 3)
        for _ in range(rng.integers(0, 2)):
            at = rng.integers(4, SIDE - 10, size=2)
            cv2.rectangle(
                image,
                tuple(at),
                tuple(at + rng.integers(3, 14, size=2)),
                tuple(int(c) for c in rng.integers(0, 200, size=3)),
                -1,
            )
        p, _ = project(keypoints[obj], r, t, K)
        records.append((image, obj, p, K, r, t, f"sim-{index}", str(index // 30)))
    return records


def load_real(keypoints):
    records = []
    with ZipFile(ROOT / "datasets/tudl_test_bop19.zip") as archive:
        files = set(archive.namelist())
        for path in sorted(files):
            if not path.endswith("/scene_gt.json"):
                continue
            directory = path.rsplit("/", 1)[0]
            ground_truth = json.loads(archive.read(path))
            information = json.loads(archive.read(directory + "/scene_gt_info.json"))
            cameras = json.loads(archive.read(directory + "/scene_camera.json"))
            for index, frame in enumerate(sorted(ground_truth, key=int)):
                suffix = f"{int(frame):06}"
                filename = next(
                    (
                        f"{directory}/rgb/{suffix}.{ext}"
                        for ext in ("png", "jpg")
                        if f"{directory}/rgb/{suffix}.{ext}" in files
                    ),
                    None,
                )
                if filename is None:
                    continue
                with Image.open(io.BytesIO(archive.read(filename))) as original:
                    original = original.convert("RGB")
                    for i, metadata in enumerate(ground_truth[frame]):
                        obj = metadata["obj_id"] - 1
                        if not 0 <= obj < len(keypoints):
                            continue
                        evidence = information[frame][i]
                        if evidence.get("visib_fract", 1.0) < 0.4:
                            continue
                        left, top, width, height = evidence["bbox_obj"]
                        if min(width, height) <= 1:
                            continue
                        side = int(np.ceil(1.35 * max(width, height)))
                        x0 = int(left + width / 2 - side / 2)
                        y0 = int(top + height / 2 - side / 2)
                        crop = original.crop((x0, y0, x0 + side, y0 + side)).resize(
                            (SIDE, SIDE), Image.Resampling.BILINEAR
                        )
                        K = np.asarray(cameras[frame]["cam_K"]).reshape(3, 3)
                        scale = SIDE / side
                        affine = np.array(
                            [
                                [scale, 0, -x0 * scale],
                                [0, scale, -y0 * scale],
                                [0, 0, 1],
                            ]
                        )
                        K = affine @ K
                        rotation = np.asarray(metadata["cam_R_m2c"]).reshape(3, 3)
                        translation = np.asarray(metadata["cam_t_m2c"])
                        coords, _ = project(keypoints[obj], rotation, translation, K)
                        records.append(
                            (
                                np.asarray(crop),
                                obj,
                                coords,
                                K,
                                rotation,
                                translation,
                                f"{directory}/{frame}/{i}",
                                f"{directory}-{index // 20}",
                            )
                        )
    if len(records) < 180:
        raise ValueError(f"Only {len(records)} annotated TUD-L detections")
    return records


def conv(a, b, stride=1):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, stride=stride, padding=1), nn.GroupNorm(8, b), nn.SiLU()
    )


class Correspondence(nn.Module):
    def __init__(self, xyz, encoded):
        super().__init__()
        self.encoded = encoded
        self.register_buffer("points", torch.from_numpy(xyz))
        distances = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
        self.register_buffer(
            "knn", torch.from_numpy(np.argsort(distances, axis=-1)[:, :, 1:7])
        )
        self.encoder = nn.Sequential(
            conv(3, 32, 2),
            conv(32, 48, 2),
            conv(48, 64, 2),
            conv(64, 96, 2),
            conv(96, 128, 2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512, 128),
        )
        self.context = nn.Sequential(nn.Linear(131, 128), nn.LayerNorm(128), nn.SiLU())
        self.message = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128)
        )
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2 * BITS + 1 if encoded else 3),
        )

    def forward(self, image, obj):
        img = self.encoder(image)
        node = self.context(
            torch.cat([self.points[obj], img[:, None].expand(-1, NODES, -1)], dim=2)
        )
        index = torch.arange(len(obj), device=obj.device)[:, None, None]
        neighbors = node[index, self.knn[obj]]
        central = node[:, :, None].expand_as(neighbors)
        edge = self.message(torch.cat([central, neighbors - central], dim=3))
        return self.output(torch.cat([node, edge.max(dim=2).values], dim=2))


def loss_and_decode(prediction, points, encoded):
    covered = ((points >= 0) & (points < SIDE)).all(dim=2)
    visibility = F.binary_cross_entropy_with_logits(
        prediction[:, :, -1], covered.float()
    )
    if encoded:
        shifts = torch.arange(BITS - 1, -1, -1, device=prediction.device)
        channels = prediction[:, :, :-1].reshape(-1, NODES, 2, BITS)
        labels = (
            (points.floor().long().clamp(0, SIDE - 1)[..., None] >> shifts) & 1
        ).float()
        loss = F.binary_cross_entropy_with_logits(
            channels[covered], labels[covered], reduction="none"
        )
        objective = (loss * torch.linspace(1.6, 0.8, BITS, device=loss.device)).mean()
        value = (((channels >= 0).long() << shifts).sum(dim=-1)).float() + 0.5
    else:
        value = prediction[:, :, :2].sigmoid() * SIDE
        objective = F.smooth_l1_loss(value[covered] / SIDE, points[covered] / SIDE)
    return objective + 0.14 * visibility, value


def splits(records, seed):
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray([row[-1] for row in records])
    stages = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval, test = next(stages.split(groups, groups=groups))
    fit, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=seed + 1).split(
            trainval, groups=groups[trainval]
        )
    )
    output = (trainval[fit], trainval[validation], test)
    assert sum(len(np.unique(groups[k])) for k in output) == len(np.unique(groups))
    return output


def test_pose(model, features, chosen, metadata, keypoints):
    images, labels, coords = features
    predicted, visible = [], []
    model.eval()
    with torch.inference_mode():
        for indexes in np.array_split(chosen, max(1, len(chosen) // 60)):
            p = model(images[indexes].float() / 255, labels[indexes])
            _, estimate = loss_and_decode(p, coords[indexes], model.encoded)
            predicted.extend(estimate.numpy())
            visible.extend(p[..., -1].sigmoid().numpy())
    pairs, rotation, translation, pixel = [], [], [], []
    cv2.setRNGSeed(22)
    for index, p, score in zip(chosen, predicted, visible):
        _, obj, actual, K, ref_r, ref_t, _, _ = metadata[index]
        truth_visible = ((actual >= 0) & (actual < SIDE)).all(axis=1)
        pixel.extend(np.linalg.norm(actual[truth_visible] - p[truth_visible], axis=1))
        nodes = np.argsort(-score)[:24]
        ok, rot, trans, _ = cv2.solvePnPRansac(
            np.ascontiguousarray(keypoints[obj][nodes]),
            np.ascontiguousarray(p[nodes]),
            K.astype(np.float64),
            None,
            iterationsCount=100,
            reprojectionError=5.0,
        )
        if ok:
            aligned_r = cv2.Rodrigues(rot)[0]
            angular = (
                np.arccos(np.clip((np.trace(aligned_r @ ref_r.T) - 1) / 2, -1, 1))
                * 180
                / np.pi
            )
            rotation.append(angular)
            translation.append(float(np.linalg.norm(trans.ravel() - ref_t)))
        pairs.append(p)
    return {
        "mean_keypoint_error_px": float(np.mean(pixel)),
        "pnp_solved_fraction": len(rotation) / len(chosen),
        "median_rotation_error_deg_for_solved": float(np.median(rotation))
        if rotation
        else None,
        "median_translation_error_mm_for_solved": float(np.median(translation))
        if translation
        else None,
    }, np.asarray(pairs)


def main(args):
    torch.set_num_threads(args.threads)
    vertices, color, points = load_models()
    rows = (
        load_real(points)
        if args.real
        else simulation(args.seed, vertices, color, points)
    )
    images = torch.from_numpy(np.stack([r[0] for r in rows])).permute(0, 3, 1, 2)
    labels = torch.tensor([r[1] for r in rows])
    xy = torch.from_numpy(np.stack([r[2] for r in rows]).astype(np.float32))
    features = images, labels, xy
    train, val, test = splits(rows, args.seed)
    output = ROOT / (
        "real-data-projects/lian-tudl-cad-correspondences"
        if args.real
        else "professor-aligned/lian-progressive-pose-codes"
    )
    output.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([r[-2] for r in rows])
    np.savez_compressed(
        output / "splits.npz",
        train=ids[train],
        validation=ids[val],
        test=ids[test],
        keypoints=points,
    )
    distance = np.linalg.norm(points, axis=2).max(axis=1)
    local = (points / distance[:, None, None]).astype(np.float32)
    report = {
        "configuration": vars(args),
        "records": len(ids),
        "held_out_frames": len(test),
        "models": {},
    }
    for name, encoded in (("regression", False), ("progressive_codes", True)):
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = Correspondence(local, encoded)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0014, weight_decay=0.003)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        history, best = [], (float("inf"), 0)
        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            model.train()
            for index in np.array_split(
                rng.permutation(train), max(1, len(train) // 40)
            ):
                pixels = images[index].float() / 255
                gain = 0.85 + 0.3 * torch.rand((len(index), 1, 1, 1))
                image = (pixels * gain).clamp(0, 1)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_and_decode(
                    model(image, labels[index]), xy[index], encoded
                )
                loss.backward()
                optimizer.step()
            schedule.step()
            accuracy, _ = test_pose(model, features, val, rows, points)
            history.append(accuracy)
            if accuracy["mean_keypoint_error_px"] < best[0]:
                best = (accuracy["mean_keypoint_error_px"], epoch)
                torch.save(model.state_dict(), output / f"{name}.pt")
            print(name, epoch, accuracy, round(time.monotonic() - start, 1), flush=True)
            if epoch - best[1] >= 4:
                break
        model.load_state_dict(torch.load(output / f"{name}.pt", weights_only=True))
        test_report, positions = test_pose(model, features, test, rows, points)
        np.savez_compressed(
            output / f"{name}-test.npz",
            predictions=positions,
            reference=xy[test].numpy(),
        )
        report["models"][name] = {
            "validation": history,
            "selected_epoch": best[1],
            "test": test_report,
        }
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=68)
    main(parser.parse_args())
