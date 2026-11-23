from __future__ import annotations

from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


def discover_pairs(root: Path):
    images = {p.stem: p for p in (root / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    masks = {p.stem: p for p in (root / "masks").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    keys = sorted(images.keys() & masks.keys())
    if not keys:
        raise FileNotFoundError(f"No matched image/mask pairs under {root}")
    return [(images[k], masks[k]) for k in keys]


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
