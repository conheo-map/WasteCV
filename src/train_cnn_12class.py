"""
train_cnn_12class.py — Huấn luyện Mô hình Phân loại Rác 12 Lớp SOTA (Production-Grade)
Tuân thủ nghiêm ngặt giao thức của 2 skill:
  - ai-model-architect: AMP (autocast + GradScaler), Gradient Clipping, Zero Memory Leak.
  - ml-data-auditor: Đánh giá độc lập trên Test Set, Macro F1, Confusion Matrix 12x12.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models

BASE_DIR = Path(__file__).resolve().parent.parent
DATA12_DIR = BASE_DIR / "data" / "12class"
TRAIN_DIR = DATA12_DIR / "train"
VAL_DIR = DATA12_DIR / "val"
TEST_DIR = DATA12_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODELS_DIR / "best_waste12_cnn.pt"


def get_data_loaders(batch_size=32):
    """Xây dựng DataLoaders với Data Augmentation chuẩn mực."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_transforms)

    # Windows an toàn với num_workers=0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes):
    """Kiến trúc MobileNetV2 Deep CNN với Custom Classifier Head 12 lớp."""
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model


def train_model(epochs=12, batch_size=32, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Bắt đầu huấn luyện Mô hình 12 Lớp trên thiết bị: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=batch_size)
    num_classes = len(classes)
    print(f"[*] Danh sách {num_classes} lớp: {classes}")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── 1. TRAIN PASS ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100

        # ── 2. VALIDATION PASS ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(enabled=use_amp, dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:2d}/{epochs:2d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": best_val_acc,
                "epoch": epoch
            }, MODEL_SAVE_PATH)
            print(f"  --> [★] Lưu checkpoint tốt nhất tại Epoch {epoch} (Val Acc: {best_val_acc:.2f}%)")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_duration = (time.time() - start_time) / 60
    print(f"\n[✓] Huấn luyện hoàn tất trong {train_duration:.2f} phút! Best Val Acc: {best_val_acc:.2f}%")

    # ── 3. TEST EVALUATION (ml-data-auditor rigor) ──
    print("\n[*] Đang đánh giá độc lập trên tập Test (Test Set Evaluation)...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = (all_preds == all_targets).mean() * 100

    report_str = classification_report(all_targets, all_preds, target_names=classes, digits=4)
    print(f"\n[✓] Test Accuracy: {test_acc:.2f}%\n")
    print(report_str)

    # Lưu Classification Report
    with open(REPORTS_DIR / "waste12_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"WasteCV 12-Class SOTA Deep CNN Model\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Training Time: {train_duration:.2f} min\n\n")
        f.write(report_str)

    # Vẽ Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="darkorange", lw=2)
    ax1.set_title("12-Class Cross-Entropy Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(history["train_acc"], label="Train Accuracy", color="royalblue", lw=2)
    ax2.plot(history["val_acc"], label="Val Accuracy", color="forestgreen", lw=2)
    ax2.set_title("12-Class Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "waste12_learning_curves.png", dpi=300)
    plt.close()

    # Vẽ Confusion Matrix 12x12
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"12-Class Confusion Matrix (Test Set Acc: {test_acc:.2f}%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "waste12_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ tại: {REPORTS_DIR / 'waste12_confusion_matrix.png'}")


if __name__ == "__main__":
    train_model(epochs=8, batch_size=64, lr=4e-4)
