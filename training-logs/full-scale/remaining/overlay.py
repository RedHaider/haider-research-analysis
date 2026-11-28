import json
from zipfile import ZipFile

import cv2
import numpy as np
from common import DATA, save, split
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

POINTS = np.array(
    [[x, y, z] for z in (-18, 18) for y in (-35, 0, 35) for x in (-35, 0, 35)],
    dtype=float,
)


def references(real, rng):
    if not real:
        for i in range(1200):
            f = rng.uniform(450, 900)
            yield (
                str(i),
                str(i),
                np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]]),
                rng.normal(0, 0.25, 3),
                np.array([*rng.uniform(-40, 40, 2), rng.uniform(400, 850)]),
            )
        return
    with ZipFile(DATA / "tudl_test_bop19.zip") as archive:
        for entry in sorted(
            k for k in archive.namelist() if k.endswith("/scene_gt.json")
        ):
            parent = entry.rsplit("/", 1)[0]
            ground = json.loads(archive.read(entry))
            intrinsics = json.loads(archive.read(parent + "/scene_camera.json"))
            for i, frame in enumerate(sorted(ground, key=int)):
                if not ground[frame]:
                    continue
                pose = ground[frame][0]
                r = Rotation.from_matrix(
                    np.asarray(pose["cam_R_m2c"]).reshape(3, 3)
                ).as_rotvec()
                K = np.asarray(intrinsics[frame]["cam_K"]).reshape(3, 3)
                yield (
                    f"{parent}/{frame}",
                    f"{parent}/{i // 12}",
                    K,
                    r,
                    np.asarray(pose["cam_t_m2c"]),
                )


def project(r, t, K):
    return cv2.projectPoints(POINTS, r, t, K, None)[0].reshape(-1, 2)


def run(real):
    rng = np.random.default_rng(82)
    rows = []
    for uid, group, K, r, t in references(real, rng):
        expected = project(r, t, K)
        for _ in range(4):
            light = rng.uniform(0.1, 1)
            confidence = rng.uniform(0.1, 1, len(POINTS))
            position = (expected - K[:2, 2]) / 320
            n = np.column_stack(
                [confidence, position, np.repeat(light, len(POINTS)), POINTS[:, 2]]
            )
            sd = np.column_stack(
                [
                    0.6 + 3.2 * (1 - confidence) ** 2 + 1.2 * abs(position[:, 0]),
                    1.0 + 4 * (1 - confidence) ** 2 + 3.0 * (1 - light) ** 2,
                ]
            )
            obs = expected + rng.normal(size=expected.shape) * sd
            rows.append((uid, group, K, expected, n, obs))
    groups = np.asarray([r[1] for r in rows])
    train, test = split(groups, groups)
    feat = np.concatenate([rows[i][4] for i in train])
    residual = np.concatenate([(rows[i][5] - rows[i][3]) ** 2 for i in train])
    fits = [
        HistGradientBoostingRegressor(
            max_iter=110, max_leaf_nodes=23, loss="poisson", random_state=16
        ).fit(feat, residual[:, axis])
        for axis in range(2)
    ]
    estimates = {"unweighted": [], "learned_precision": [], "robust_precision": []}
    for index in test:
        _, _, K, expected, n, obs = rows[index]
        ok, r, t = cv2.solvePnP(POINTS, obs, K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError("Initial PnP did not converge")
        init = np.r_[r.ravel(), t.ravel()]
        weights = 1 / np.sqrt(
            np.column_stack([fit.predict(n) for fit in fits]).clip(0.18, 100)
        )
        for name, weight, loss in (
            ("unweighted", 1.0, "linear"),
            ("learned_precision", weights, "linear"),
            ("robust_precision", weights, "soft_l1"),
        ):
            fit = least_squares(
                lambda q, K=K, obs=obs, weight=weight: (
                    (project(q[:3], q[3:], K) - obs) * weight
                ).ravel(),
                init,
                loss=loss,
                max_nfev=24,
            )
            estimates[name].append(
                float(
                    np.linalg.norm(
                        project(fit.x[:3], fit.x[3:], K) - expected, axis=1
                    ).mean()
                )
            )
    report = {
        "reference": "TUD-L camera intrinsics and measured object poses"
        if real
        else "simulated cameras and poses",
        "observation_noise": "injected anisotropic Gaussian error",
        "training_instances": len(train),
        "test_instances": len(test),
        "held_out_corner_error_px": {
            name: {
                "median": float(np.median(v)),
                "mean": float(np.mean(v)),
                "p90": float(np.quantile(v, 0.9)),
            }
            for name, v in estimates.items()
        },
    }
    save(
        "azimi-tudl-overlay-calibration"
        if real
        else "azimi-anisotropic-hmd-calibration",
        real,
        report,
        fits,
        {
            "train_groups": groups[train],
            "test_groups": groups[test],
            **{n: np.array(v) for n, v in estimates.items()},
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            run(real)
