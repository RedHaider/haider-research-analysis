import cv2
import numpy as np
from common import save, split
from plugplay import load_slices
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

N = 32


def source(real, r):
    if real:
        return load_slices("task02-heart.tar")
    z, y = np.mgrid[-1 : 1 : complex(N), -1 : 1 : complex(N)]
    images = []
    for _ in range(1600):
        r1, r2 = r.uniform(0.25, 0.75, 2)
        ring = (z / r1) ** 2 + (y / r2) ** 2
        x = 0.6 * np.exp(-((ring - 1) ** 2) * 3) + 0.26 * gaussian_filter(
            r.random((N, N)), 0.75
        )
        images.append(x)
    return np.asarray(images, dtype=np.float32), np.arange(len(images))


def frames(images, trials, rng):
    yy, xx = np.mgrid[:N, :N]
    for im in images:
        for _ in range(trials):
            a, b = rng.uniform(-2.5, 2.5, 2)
            amp = rng.uniform(0, 1.8)
            dx = a + amp * np.sin(yy / 12 + rng.uniform(0, 6))
            dy = b + amp * np.cos(xx / 11 + rng.uniform(0, 6))
            target = np.stack([dx, dy], axis=-1).astype(np.float32)
            moved = map_coordinates(im, [yy + dy, xx + dx], order=1, mode="reflect")
            sx, sy = rng.uniform(0.3, 1.6, 2)
            fixed = gaussian_filter(im, (sy, sx))
            moved = gaussian_filter(moved, (sy * 0.72, sx * 1.18))
            fixed = np.clip(fixed + rng.normal(0, 0.026, fixed.shape), 0, 1)
            moved = np.clip(moved + rng.normal(0, 0.026, moved.shape), 0, 1)
            # Moving-to-fixed flow is defined at moving-image coordinates.
            u = np.ascontiguousarray(moved * 255, dtype=np.uint8)
            v = np.ascontiguousarray(fixed * 255, dtype=np.uint8)
            flows = [
                cv2.calcOpticalFlowFarneback(u, v, None, 0.5, 2, w, 3, 5, 1.2, 0)
                for w in (7, 13, 21)
            ]
            feature = np.column_stack(
                [
                    *[(f[4:-4, 4:-4].reshape(-1, 2)) for f in flows],
                    moved[4:-4, 4:-4].ravel(),
                    fixed[4:-4, 4:-4].ravel(),
                    yy[4:-4, 4:-4].ravel() / N,
                    xx[4:-4, 4:-4].ravel() / N,
                    np.full((N - 8) ** 2, sx),
                    np.full((N - 8) ** 2, sy),
                ]
            )
            yield (
                feature,
                target[4:-4, 4:-4].reshape(-1, 2),
                flows[1][4:-4, 4:-4].reshape(-1, 2),
            )


def main(real):
    cv2.setNumThreads(1)
    r = np.random.default_rng(78)
    x, group = source(real, r)
    train, test = split(group, group, seed=78)
    input_x, target = [], []
    for ft, y, _ in frames(x[train], 5 if real else 1, r):
        take = r.choice(len(ft), 96, replace=False)
        input_x.append(ft[take])
        target.append(y[take])
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(192, 128, 64),
            batch_size=512,
            max_iter=40,
            early_stopping=True,
            random_state=78,
        ),
    )
    model.fit(
        np.concatenate(input_x).astype(np.float32),
        np.concatenate(target).astype(np.float32),
    )
    errors, fit = [], []
    for ft, y, b in frames(x[test], 8 if real else 2, r):
        p = model.predict(ft.astype(np.float32))
        errors.append(float(np.linalg.norm(b - y, axis=1).mean()))
        fit.append(float(np.linalg.norm(p - y, axis=1).mean()))
    save(
        "amini-cardiac-mri-motion" if real else "amini-psf-motion-estimation",
        real,
        {
            "input": "MSD heart MRI" if real else "cardiac-like phantoms",
            "motion": "injected displacement and anisotropic blur",
            "source_images": len(x),
            "training_volumes": len(np.unique(group[train])),
            "held_out_volumes": len(np.unique(group[test])),
            "test_pairs": len(fit),
            "mean_endpoint_error_px": {
                "farneback": float(np.mean(errors)),
                "psf_conditioned": float(np.mean(fit)),
            },
        },
        model,
        {
            "train_groups": group[train],
            "test_groups": group[test],
            "farneback": np.asarray(errors),
            "learned": np.asarray(fit),
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
