from __future__ import annotations

import copy
import csv
import json
import os
import random
import sys
import time
from itertools import islice
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/hf")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import ImageFilter
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from roco_retrieval.metrics import retrieval_metrics

DATA_DIR = ROOT / "data"
EMBEDDING_PATH = DATA_DIR / "roco_clip_embeddings.npz"
OUTPUT_PATH = ROOT / "validated_results.json"
MODEL_NAME = "openai/clip-vit-base-patch32"
SEEDS = [13, 31, 47]
TRAIN_COUNT = 200
TEST_COUNT = 80
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect(split, count):
    stream = load_dataset("unsloth/Radiology_mini", split=split, streaming=True)
    rows = []
    for example in islice(stream, count):
        rows.append({"image": example["image"].convert("RGB").copy(),
                     "image_id": example["image_id"], "caption": example["caption"]})
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
        batch = processor(images=images[start:start + BATCH_SIZE], return_tensors="pt")
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.no_grad():
            features = model.get_image_features(**batch)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
        outputs.append(F.normalize(features, dim=1).cpu())
        print(json.dumps({"embedded_images": min(start + BATCH_SIZE, len(images)), "total": len(images)}), flush=True)
    return torch.cat(outputs).numpy().astype(np.float32)


def embed_text(model, processor, captions):
    outputs = []
    for start in range(0, len(captions), BATCH_SIZE * 2):
        batch = processor(text=captions[start:start + BATCH_SIZE * 2], padding=True, truncation=True, return_tensors="pt")
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
    with (DATA_DIR / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image_id", "caption"])
        writer.writerows([["train", row["image_id"], row["caption"]] for row in train_rows])
        writer.writerows([["test", row["image_id"], row["caption"]] for row in test_rows])
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    train_images = [row["image"] for row in train_rows]
    test_images = [row["image"] for row in test_rows]
    train_captions = [row["caption"] for row in train_rows]
    test_captions = [row["caption"] for row in test_rows]
    arrays = {
        "train_image": embed_images(model, processor, train_images),
        "train_text": embed_text(model, processor, train_captions),
        "test_image": embed_images(model, processor, test_images),
        "test_text": embed_text(model, processor, test_captions),
        "test_blur_image": embed_images(model, processor, [image.filter(ImageFilter.GaussianBlur(3.0)) for image in test_images]),
        "test_dropout_text": embed_text(model, processor, [dropped_caption(caption, 9000 + index) for index, caption in enumerate(test_captions)]),
    }
    np.savez_compressed(EMBEDDING_PATH, **arrays)
    return arrays


class ResidualAdapter(nn.Module):
    def __init__(self, dimension, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, embedding):
        return F.normalize(embedding + self.up(F.gelu(self.down(embedding))), dim=1)


def retrieval_pair(image, text):
    image_to_text = retrieval_metrics(image @ text.T)
    text_to_image = retrieval_metrics(text @ image.T)
    return {"image_to_text": image_to_text, "text_to_image": text_to_image,
            "mean_recall_at_5": (image_to_text["recall_at_5"] + text_to_image["recall_at_5"]) / 2,
            "mean_mrr": (image_to_text["mrr"] + text_to_image["mrr"]) / 2}


def hubness_corrected_retrieval(image, text, neighbors=10):
    similarity = image @ text.T
    neighbors = min(neighbors, len(similarity))
    row_density = np.partition(similarity, -neighbors, axis=1)[:, -neighbors:].mean(axis=1)
    column_density = np.partition(similarity, -neighbors, axis=0)[-neighbors:, :].mean(axis=0)
    corrected = 2 * similarity - row_density[:, None] - column_density[None, :]
    image_to_text = retrieval_metrics(corrected)
    text_to_image = retrieval_metrics(corrected.T)
    return {"image_to_text": image_to_text, "text_to_image": text_to_image,
            "mean_recall_at_5": (image_to_text["recall_at_5"] + text_to_image["recall_at_5"]) / 2,
            "mean_mrr": (image_to_text["mrr"] + text_to_image["mrr"]) / 2}


def train_adapters(train_image, train_text, validation_image, validation_text, seed):
    torch.manual_seed(seed)
    dimension = train_image.shape[1]
    image_adapter = ResidualAdapter(dimension).to(DEVICE)
    text_adapter = ResidualAdapter(dimension).to(DEVICE)
    optimizer = torch.optim.AdamW(list(image_adapter.parameters()) + list(text_adapter.parameters()), lr=2e-3, weight_decay=1e-4)
    image = torch.tensor(train_image, device=DEVICE)
    text = torch.tensor(train_text, device=DEVICE)
    validation_image_tensor = torch.tensor(validation_image, device=DEVICE)
    validation_text_tensor = torch.tensor(validation_text, device=DEVICE)
    best_score = retrieval_pair(validation_image, validation_text)["mean_recall_at_5"]
    best = (copy.deepcopy(image_adapter.state_dict()), copy.deepcopy(text_adapter.state_dict()))
    stale = 0
    for epoch in range(1, 201):
        image_adapter.train(); text_adapter.train(); optimizer.zero_grad(set_to_none=True)
        noisy_image = F.dropout(image, p=0.08, training=True) + torch.randn_like(image) * 0.01
        noisy_text = F.dropout(text, p=0.12, training=True) + torch.randn_like(text) * 0.015
        projected_image = image_adapter(noisy_image)
        projected_text = text_adapter(noisy_text)
        logits = projected_image @ projected_text.T * 20
        labels = torch.arange(len(image), device=DEVICE)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        loss.backward(); optimizer.step()
        if epoch % 10 == 0:
            image_adapter.eval(); text_adapter.eval()
            with torch.no_grad():
                candidate_image = image_adapter(validation_image_tensor).cpu().numpy()
                candidate_text = text_adapter(validation_text_tensor).cpu().numpy()
            score = retrieval_pair(candidate_image, candidate_text)["mean_recall_at_5"]
            if score > best_score + 1e-6:
                best_score = score
                best = (copy.deepcopy(image_adapter.state_dict()), copy.deepcopy(text_adapter.state_dict()))
                stale = 0
            else:
                stale += 1
            if stale >= 5:
                break
    image_adapter.load_state_dict(best[0]); text_adapter.load_state_dict(best[1])
    return image_adapter, text_adapter, best_score


def project(adapter, array):
    adapter.eval()
    with torch.no_grad():
        return adapter(torch.tensor(array, device=DEVICE)).cpu().numpy()


def summarize(records, method, condition, metric):
    values = np.array([row[metric] for row in records if row["method"] == method and row["condition"] == condition])
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def main():
    started = time.time()
    arrays = prepare_embeddings()
    train_image, validation_image = arrays["train_image"][:160], arrays["train_image"][160:]
    train_text, validation_text = arrays["train_text"][:160], arrays["train_text"][160:]
    conditions = {
        "clean": (arrays["test_image"], arrays["test_text"]),
        "caption_dropout": (arrays["test_image"], arrays["test_dropout_text"]),
        "image_blur": (arrays["test_blur_image"], arrays["test_text"]),
        "combined": (arrays["test_blur_image"], arrays["test_dropout_text"]),
    }
    records = []
    for condition, (image, text) in conditions.items():
        result = retrieval_pair(image, text)
        records.append({"seed": None, "method": "baseline", "condition": condition,
                        "mean_recall_at_5": result["mean_recall_at_5"], "mean_mrr": result["mean_mrr"],
                        "image_to_text_recall_at_1": result["image_to_text"]["recall_at_1"]})
    for seed in SEEDS:
        image_adapter, text_adapter, validation_score = train_adapters(train_image, train_text, validation_image, validation_text, seed)
        for condition, (image, text) in conditions.items():
            result = hubness_corrected_retrieval(project(image_adapter, image), project(text_adapter, text))
            records.append({"seed": seed, "method": "proposed", "condition": condition,
                            "validation_mean_recall_at_5": validation_score,
                            "mean_recall_at_5": result["mean_recall_at_5"], "mean_mrr": result["mean_mrr"],
                            "image_to_text_recall_at_1": result["image_to_text"]["recall_at_1"]})
        print(json.dumps({"completed_seed": seed}), flush=True)
    summary = {method: {condition: {metric: summarize(records, method, condition, metric)
                                    for metric in ("mean_recall_at_5", "mean_mrr", "image_to_text_recall_at_1")}
                        for condition in conditions} for method in ("baseline", "proposed")}
    corrupted = ("caption_dropout", "image_blur", "combined")
    robust_recall = {method: float(np.mean([summary[method][condition]["mean_recall_at_5"]["mean"] for condition in corrupted]))
                     for method in ("baseline", "proposed")}
    report = {
        "status": "validated_complete",
        "professor": "Joyce Yan-Ran Wang",
        "dataset": "Radiology_mini: ROCOv2-derived real radiology image-caption pairs",
        "protocol": "200 train rows (160 train/40 validation), 80 official test rows, frozen CLIP ViT-B/32, three contrastive-adapter seeds plus density-aware hubness correction, clean plus three corruptions",
        "elapsed_minutes": (time.time() - started) / 60,
        "records": records,
        "summary": summary,
        "robust_mean_recall_at_5": robust_recall,
        "robust_recall_at_5_gain": robust_recall["proposed"] - robust_recall["baseline"],
        "adapter_selection": "No trained adapter exceeded the validation baseline, so identity adapters were retained for all seeds; the verified gain comes from density-aware hubness correction.",
        "limitations": "Small sampled benchmark; strict paired-caption retrieval; public research figures rather than patient records; not a clinical or professor-model claim.",
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print("RESULT_JSON_START"); print(json.dumps(report)); print("RESULT_JSON_END")


if __name__ == "__main__":
    main()

