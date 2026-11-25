from __future__ import annotations


import csv
import json
import os
import random


from itertools import islice
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parent / "data" / "hf_cache")
)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch

import torch.nn.functional as F
from datasets import load_dataset
from PIL import ImageFilter
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
EMBEDDING_PATH = DATA_DIR / "roco_clip_embeddings.npz"
MODEL_NAME = "openai/clip-vit-base-patch32"
TRAIN_COUNT = 200
TEST_COUNT = 80
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect(split, count):
    stream = load_dataset(
        "unsloth/Radiology_mini",
        split=split,
        streaming=True,
        revision="12468bc1f3cecf1056c290182a0e88ce506049f9",
    )
    rows = []
    for example in islice(stream, count):
        rows.append(
            {
                "image": example["image"].convert("RGB").copy(),
                "image_id": example["image_id"],
                "caption": example["caption"],
            }
        )
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} {split} rows, received {len(rows)}")
    return rows


def dropped_caption(caption, seed):
    words = caption.split()
    if len(words) <= 3:
        return caption
    rng = random.Random(seed)
    kept = [word for word in words if rng.random() > 0.35]
    return " ".join(kept if len(kept) >= 3 else words[:3])


def embed_images(model, processor, images):
    outputs = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = processor(
            images=images[start : start + BATCH_SIZE], return_tensors="pt"
        )
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.no_grad():
            features = model.get_image_features(**batch)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
        outputs.append(F.normalize(features, dim=1).cpu())
        print(
            json.dumps(
                {
                    "embedded_images": min(start + BATCH_SIZE, len(images)),
                    "total": len(images),
                }
            ),
            flush=True,
        )
    return torch.cat(outputs).numpy().astype(np.float32)


def embed_text(model, processor, captions):
    outputs = []
    for start in range(0, len(captions), BATCH_SIZE * 2):
        batch = processor(
            text=captions[start : start + BATCH_SIZE * 2],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.no_grad():
            features = model.get_text_features(**batch)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
        outputs.append(F.normalize(features, dim=1).cpu())
    return torch.cat(outputs).numpy().astype(np.float32)


def prepare_embeddings():
    if EMBEDDING_PATH.exists():
        return dict(np.load(EMBEDDING_PATH))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = collect("train", TRAIN_COUNT)
    test_rows = collect("test", TEST_COUNT)
    with (DATA_DIR / "sample_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image_id", "caption"])
        writer.writerows(
            [["train", row["image_id"], row["caption"]] for row in train_rows]
        )
        writer.writerows(
            [["test", row["image_id"], row["caption"]] for row in test_rows]
        )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    model = (
        CLIPModel.from_pretrained(
            MODEL_NAME, revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        )
        .to(DEVICE)
        .eval()
    )
    train_images = [row["image"] for row in train_rows]
    test_images = [row["image"] for row in test_rows]
    train_captions = [row["caption"] for row in train_rows]
    test_captions = [row["caption"] for row in test_rows]
    arrays = {
        "train_image": embed_images(model, processor, train_images),
        "train_text": embed_text(model, processor, train_captions),
        "test_image": embed_images(model, processor, test_images),
        "test_text": embed_text(model, processor, test_captions),
        "test_blur_image": embed_images(
            model,
            processor,
            [image.filter(ImageFilter.GaussianBlur(3.0)) for image in test_images],
        ),
        "test_dropout_text": embed_text(
            model,
            processor,
            [
                dropped_caption(caption, 9000 + index)
                for index, caption in enumerate(test_captions)
            ],
        ),
    }
    np.savez_compressed(EMBEDDING_PATH, **arrays)
    return arrays


if __name__ == "__main__":
    prepare_embeddings()
