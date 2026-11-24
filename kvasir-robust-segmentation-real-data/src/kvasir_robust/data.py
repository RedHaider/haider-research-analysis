from __future__ import annotations

from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import hashlib


def discover_pairs(root: Path):
    images = {p.stem: p for p in (root / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    masks = {p.stem: p for p in (root / "masks").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    keys = sorted(images.keys() & masks.keys())
    if not keys:
        raise FileNotFoundError(f"No matched image/mask pairs under {root}")
    pairs = [(images[k], masks[k]) for k in keys]
    # Fail fast on corrupt exports (for example, binary {0,1} masks saved as
    # JPEG and later thresholded at 127). Kvasir-SEG contains foreground in
    # every ground-truth mask.
    empty, non_binary = [], []
    for _, mask_path in pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unreadable mask: {mask_path}")
        foreground = float((mask > 127).mean())
        if foreground == 0.0:
            empty.append(mask_path.name)
        if foreground == 1.0:
            non_binary.append(mask_path.name)
    if empty:
        raise ValueError(f"Found {len(empty)} empty masks after thresholding; first: {empty[:5]}")
    if non_binary:
        raise ValueError(f"Found {len(non_binary)} all-foreground masks; first: {non_binary[:5]}")
    return pairs


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_split_isolation(train_pairs, val_pairs, test_pairs):
    """Reject filename and byte-identical image leakage between partitions."""
    groups = {"train": train_pairs, "validation": val_pairs, "test": test_pairs}
    names = {key: {p[0].name for p in value} for key, value in groups.items()}
    digests = {key: {file_digest(p[0]) for p in value} for key, value in groups.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if names[left] & names[right]:
            raise ValueError(f"Filename leakage between {left} and {right}")
        if digests[left] & digests[right]:
            raise ValueError(f"Byte-identical image leakage between {left} and {right}")


def fold_indices(n: int, folds: int, seed: int, fold: int):
    splits = list(KFold(folds, shuffle=True, random_state=seed).split(np.arange(n)))
    train_all, test = splits[fold]
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(train_all)
    val_n = max(1, int(.15 * len(train_all)))
    return train_all[val_n:], train_all[:val_n], test


def transforms(size: int, train: bool, robust: bool):
    operations = [A.Resize(size, size)]
    if train:
        operations += [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.15), A.Rotate(limit=20, p=.35),
                       A.RandomBrightnessContrast(.2, .2, p=.35)]
        if robust:
            operations += [A.OneOf([A.GaussNoise(std_range=(.03, .10)), A.GaussianBlur((3, 7)),
                                    A.ImageCompression(quality_range=(45, 85))], p=.45)]
    operations += [A.Normalize()]
    return A.Compose(operations)


class SegmentationDataset(Dataset):
    def __init__(self, pairs, transform): self.pairs, self.transform = pairs, transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        sample = self.transform(image=image, mask=(mask > 127).astype(np.float32))
        image = torch.from_numpy(sample["image"].transpose(2, 0, 1)).float()
        mask = torch.from_numpy(sample["mask"][None]).float()
        return image, mask, image_path.name

