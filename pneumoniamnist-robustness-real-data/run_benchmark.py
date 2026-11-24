from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import medmnist
import numpy as np
import torch
import torch.nn as nn
from medmnist import PneumoniaMNIST
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("D:/USA based documents/Haider Heavy Research Projects/datasets/medmnist")
OUTPUT_PATH = Path(__file__).with_name("validated_results.json")
SEEDS = [17, 29, 43]
IMAGE_SIZE = 28
EPOCHS = 6
PATIENCE = 3
BATCH = 512


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_noise(image):
    return (image + torch.randn_like(image) * 0.09).clamp(0, 1)


base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])
robust_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.35),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.Lambda(tensor_noise)], p=0.45),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_dataset(split, transform):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return PneumoniaMNIST(split=split, transform=transform, download=True, size=28, root=str(DATA_ROOT))


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = (nn.Identity() if input_channels == output_channels and stride == 1 else
                     nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                                   nn.BatchNorm2d(output_channels)))
        self.activation = nn.SiLU()

    def forward(self, image):
        return self.activation(self.body(image) + self.skip(image))


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU(),
        ResidualBlock(16, 32, 2), ResidualBlock(32, 32),
        ResidualBlock(32, 64, 2), ResidualBlock(64, 64),
        ResidualBlock(64, 128, 2), ResidualBlock(128, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 1),
    )


def expected_calibration_error(probability, target, bins=15):
    confidence = np.maximum(probability, 1 - probability)
    prediction = (probability >= 0.5).astype(int)
    correctness = (prediction == target).astype(float)
    edges = np.linspace(0.5, 1, bins + 1)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (confidence >= low) & (confidence <= high if index == bins - 1 else confidence < high)
        if keep.any():
            value += keep.mean() * abs(correctness[keep].mean() - confidence[keep].mean())
    return float(value)


def metric_set(probability, target):
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "f1": float(f1_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "ece": expected_calibration_error(probability, target),
    }


def collect_logits(model, loader):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for image, target in loader:
            logits.append(model(image.to(DEVICE)).squeeze(1).cpu())
            targets.append(target.view(-1).float())
    return torch.cat(logits), torch.cat(targets)


def fit_temperature(logits, targets):
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.5, 5))


def train_one(method, seed):
    seed_all(seed)
    train_dataset = get_dataset("train", robust_transform if method == "proposed" else base_transform)
    validation_dataset = get_dataset("val", base_transform)
    labels = train_dataset.labels.reshape(-1)
    counts = np.bincount(labels)
    weights = 1 / counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    model = make_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[0] / counts[1]], device=DEVICE))
    best_auc, best_state, stale = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for image, target in train_loader:
            image = image.to(DEVICE)
            target = target.float().view(-1, 1).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            if method == "proposed":
                target = target * 0.95 + 0.025
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        validation_logits, validation_targets = collect_logits(model, validation_loader)
        auc = roc_auc_score(validation_targets.numpy(), torch.sigmoid(validation_logits).numpy())
        print(json.dumps({"method": method, "seed": seed, "epoch": epoch, "validation_auroc": auc}), flush=True)
        if auc > best_auc + 1e-4:
            best_auc, best_state, stale = auc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    validation_logits, validation_targets = collect_logits(model, validation_loader)
    temperature = fit_temperature(validation_logits, validation_targets) if method == "proposed" else 1.0
    return model, temperature


def corruption_transform(condition):
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    if condition == "noise":
        operations.append(transforms.Lambda(lambda image: (image + torch.randn_like(image) * 0.14).clamp(0, 1)))
    elif condition == "blur":
        operations.append(transforms.GaussianBlur(7, sigma=2.0))
    elif condition == "contrast":
        operations.append(transforms.Lambda(lambda image: ((image - 0.5) * 0.45 + 0.5).clamp(0, 1)))
    elif condition == "brightness":
        operations.append(transforms.Lambda(lambda image: (image * 0.55).clamp(0, 1)))
    operations.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(operations)


def evaluate(model, temperature, condition):
    dataset = get_dataset("test", corruption_transform(condition))
    logits, targets = collect_logits(model, DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0))
    probability = torch.sigmoid(logits / temperature).numpy()
    return metric_set(probability, targets.numpy().astype(int))


def main():
    started = time.time()
    records = []
    for seed in SEEDS:
        for method in ("baseline", "proposed"):
            model, temperature = train_one(method, seed)
            for condition in ("clean", "noise", "blur", "contrast", "brightness"):
                row = {"seed": seed, "method": method, "condition": condition, "temperature": temperature,
                       **evaluate(model, temperature, condition)}
                records.append(row)
                print(json.dumps(row), flush=True)

    def summarize(method, condition, metric):
        values = [row[metric] for row in records if row["method"] == method and row["condition"] == condition]
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    metrics = ("auroc", "auprc", "f1", "balanced_accuracy", "ece")
    conditions = ("clean", "noise", "blur", "contrast", "brightness")
    summary = {method: {condition: {metric: summarize(method, condition, metric) for metric in metrics}
                        for condition in conditions} for method in ("baseline", "proposed")}
    robust_mean = {method: float(np.mean([summary[method][condition]["auroc"]["mean"]
                                         for condition in conditions[1:]]))
                   for method in ("baseline", "proposed")}
    report = {
        "status": "validated_complete",
        "professor": "Truong-Son Hy",
        "dataset": "PneumoniaMNIST 28px official splits",
        "protocol": "3 seeds; official train/validation/test; residual CNN; early stopping; clean plus four corruptions",
        "elapsed_minutes": (time.time() - started) / 60,
        "records": records,
        "summary": summary,
        "robust_mean_auroc": robust_mean,
        "robust_auroc_gain": robust_mean["proposed"] - robust_mean["baseline"],
        "limitations": "Portfolio benchmark; not a professor-model reproduction or clinical claim.",
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print("RESULT_JSON_START")
    print(json.dumps(report))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()

