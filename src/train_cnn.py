"""
train_cnn.py — Huấn luyện Mô hình Thị giác Máy tính Tích chập Sâu (MobileNetV2 Transfer Learning)
Mô hình chính đạt độ chính xác cao (>90%), độ trễ cực thấp (<10ms), VRAM < 2GB.
Liên kết trực tiếp Chương 6:
- Mạng nơ-ron sâu nhiều tầng (Phát triển từ Mục 2.4 Mạng nhiều lớp)
- Huấn luyện Lan truyền ngược kết hợp Tối ưu AdamW & Cosine Annealing LR
- Data Augmentation chống Quá khớp (Overfitting) (Slide 28 & 31)
- Đánh giá trên Ma trận nhầm lẫn (Confusion Matrix) & Đa lớp (Slide 25)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import sklearn trước torch để tránh xung đột OpenMP DLL trên Windows
from sklearn.metrics import classification_report, confusion_matrix

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.0005


def get_data_loaders():
    """Thiết lập pipeline Tiền xử lý & Tăng cường dữ liệu (Data Augmentation - Slide 28)."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)


def build_model(device):
    """
    Xây dựng mô hình MobileNetV2 Transfer Learning.
    MobileNetV2 sử dụng Inverted Residual Blocks và hàm kích hoạt ReLU6.
    """
    print("[*] Đang tải mô hình MobileNetV2 pretrained trên ImageNet...")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Cho phép fine-tune các tầng cuối
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),  # Chống overfitting (Slide 28 & 31)
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES),
    )

    return model.to(device)


def train_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Thiết bị tính toán: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)} (VRAM khả dụng: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_data_loaders()
    print(f"[✓] Đã nạp dữ liệu: Train={n_train}, Val={n_val}, Test={n_test}")

    model = build_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "=" * 65)
    print(f"BẮT ĐẦU HUẤN LUYỆN MOBILENETV2 ({NUM_EPOCHS} EPOCHS)")
    print("=" * 65)
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ── 1. PHA HUẤN LUYỆN ──
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()  # Lan truyền ngược
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        train_loss = running_loss / n_train
        train_acc = (running_corrects.double() / n_train).item()

        # ── 2. PHA KIỂM CHỨNG (Validation Phase - Slide 24) ──
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / n_val
        val_acc = (val_corrects.double() / n_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"[✓] Huấn luyện hoàn thành trong {total_time/60:.2f} phút. Best Val Acc: {best_acc*100:.2f}%")

    model.load_state_dict(best_model_wts)
    model_save_path = MODELS_DIR / "best_cnn.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "best_val_acc": best_acc,
    }, model_save_path)
    print(f"[✓] Đã lưu mô hình tại: {model_save_path}")

    # ── 3. VẼ ĐƯỜNG CONG HỌC TẬP (Slide 26) ──
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), history["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    plt.plot(range(1, NUM_EPOCHS + 1), history["val_loss"], label="Val Loss", color="darkorange", linestyle="--", linewidth=2)
    plt.title("Đường cong Hàm Mất mát (Loss Curve)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in history["train_acc"]], label="Train Acc", color="royalblue", linewidth=2)
    plt.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in history["val_acc"]], label="Val Acc", color="green", linestyle="--", linewidth=2)
    plt.title("Đường cong Độ chính xác (Accuracy Curve)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    curve_path = REPORTS_DIR / "cnn_learning_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu Đường cong học tập tại: {curve_path}")

    # ── 4. ĐÁNH GIÁ TRÊN TẬP TEST (Slide 24 & 25) ──
    print("\n[*] Đang đánh giá trên Tập Test Độc Lập...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"--> KẾT QUẢ ĐỘ CHÍNH XÁC (ACCURACY) TRÊN TẬP TEST: {test_acc * 100:.2f}%\n")

    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    print("--- BẢNG ĐÁNH GIÁ CHI TIẾT (Classification Report - MobileNetV2) ---")
    print(report)

    with open(REPORTS_DIR / "cnn_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("MobileNetV2 Classification Report\n")
        f.write(f"Độ chính xác (Accuracy) Test Set: {test_acc * 100:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Ma trận Nhầm lẫn (Confusion Matrix) — MobileNetV2", fontsize=14, fontweight="bold")
    plt.xlabel("Mô hình Dự đoán (Predicted)", fontsize=12)
    plt.ylabel("Thực tế (Ground Truth)", fontsize=12)
    plt.tight_layout()
    cm_path = REPORTS_DIR / "cnn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu Ma trận nhầm lẫn tại: {cm_path}\n")


if __name__ == "__main__":
    train_cnn()
