import gzip
import tarfile

import nibabel as nib
import numpy as np
from common import DATA, save, split
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, median_filter, zoom
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

SIZE = 32


def load_slices(archive):
    out, g = [], []
    with tarfile.open(DATA / archive) as tar:
        for member in tar:
            name = member.name
            if not (
                "/imagesTr/" in name
                and name.endswith(".nii.gz")
                and not name.rsplit("/", 1)[-1].startswith("._")
            ):
                continue
            with tar.extractfile(member) as f:
                a = nib.Nifti1Image.from_bytes(gzip.decompress(f.read())).get_fdata(
                    dtype=np.float32
                )
            a = np.nan_to_num(a)
            for i in np.linspace(0.26, 0.72, 7):
                x = a[:, :, int(i * a.shape[2])]
                lo, hi = np.percentile(x, (2, 98))
                if hi - lo < 1e-5:
                    continue
                x = np.clip((x - lo) / (hi - lo), 0, 1)
                out.append(zoom(x, (SIZE / x.shape[0], SIZE / x.shape[1]), order=1))
                g.append(name)
    return np.stack(out).astype(np.float32), np.asarray(g)


def source(real):
    if real:
        return load_slices("msd-hippocampus.tar")
    rng = np.random.default_rng(59)
    a, b = np.mgrid[-1 : 1 : complex(SIZE), -1 : 1 : complex(SIZE)]
    out = []
    for _ in range(2600):
        x = np.zeros((SIZE, SIZE))
        for i in range(rng.integers(3, 8)):
            shift = rng.uniform(-0.7, 0.7, 2)
            scale = rng.uniform(0.1, 0.65, 2)
            e = ((a - shift[0]) / scale[0]) ** 2 + ((b - shift[1]) / scale[1]) ** 2
            x += (e < 1) * rng.uniform(0.18, 0.65)
        out.append(gaussian_filter(np.clip(x, 0, 1), rng.uniform(0.4, 1.3)))
    return np.stack(out).astype(np.float32), np.arange(len(out))


def patches(x):
    padded = np.pad(x, ((0, 0), (2, 2), (2, 2)), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, (5, 5), axis=(-2, -1))


def main(real):
    rng = np.random.default_rng(59)
    clean, groups = source(real)
    train, test = split(groups, groups)
    noise = rng.uniform(0.025, 0.14, (len(train), 1, 1)) * rng.normal(
        size=clean[train].shape
    )
    dirty = (clean[train] + noise).astype(np.float32)
    dense = patches(dirty)
    locations = rng.integers(0, SIZE, (len(train), 50, 2))
    i = np.arange(len(train))[:, None]
    r, c = locations[:, :, 0], locations[:, :, 1]
    fit_x = dense[i, r, c].reshape(-1, 25)
    fit_y = (clean[train][i, r, c] - dirty[i, r, c]).ravel()
    denoise = MLPRegressor(
        hidden_layer_sizes=(160, 96, 64),
        batch_size=512,
        max_iter=30,
        early_stopping=True,
        random_state=59,
    ).fit(fit_x, fit_y)
    psnr = {"zero_filled": [], "median_consistency": [], "learned_consistency": []}
    for start in range(0, len(test), 12):
        target = clean[test[start : start + 12]]
        shape = target.shape
        mask = rng.random(shape) < 0.22
        mask[:, :6, :6] = True
        coeff = dctn(target, axes=(-2, -1), norm="ortho")
        sampled = coeff + rng.normal(0, 0.023, shape)
        base = idctn(np.where(mask, sampled, 0), axes=(-2, -1), norm="ortho")
        for name, scores in psnr.items():
            image = base.copy()
            if name != "zero_filled":
                for _ in range(4):
                    if name == "learned_consistency":
                        correction = denoise.predict(
                            patches(image).reshape(-1, 25).astype(np.float32)
                        ).reshape(shape)
                        smooth = (image + 0.7 * correction).clip(0, 1)
                    else:
                        smooth = median_filter(image, size=(1, 3, 3))
                    mat = dctn(smooth, axes=(-2, -1), norm="ortho")
                    image = idctn(
                        np.where(mask, sampled, mat), axes=(-2, -1), norm="ortho"
                    )
            scores.extend(
                (
                    -10
                    * np.log10(
                        np.mean((image.clip(0, 1) - target) ** 2, axis=(1, 2)).clip(
                            1e-8
                        )
                    )
                ).tolist()
            )
    save(
        "katsaggelos-mri-consistency" if real else "katsaggelos-plugplay-phantoms",
        real,
        {
            "source": "MSD hippocampus MRI slices" if real else "simulated phantoms",
            "measurement": "simulated missing cosine-transform coefficients",
            "train_images": len(train),
            "test_images": len(test),
            "train_patches": len(fit_x),
            "mean_psnr_db": {n: float(np.mean(s)) for n, s in psnr.items()},
        },
        denoise,
        {
            "train_group": groups[train],
            "test_group": groups[test],
            **{n: np.asarray(s) for n, s in psnr.items()},
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for real in (False, True):
            main(real)
