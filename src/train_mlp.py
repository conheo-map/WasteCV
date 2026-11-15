"""
train_mlp.py — Huấn luyện Mô hình Mạng Nơ-ron Nhiều Lớp (MLP Classifier Baseline)
Khớp đúng 100% nội dung Mục 2 trong Slide Chương 6:
- Mô hình Nơ-ron: Net = Σ w_j * s_j, out = g(Net) (Slide 37)
- Hàm kích hoạt phi tuyến: ReLU (Slide 38)
- Giải thuật Lan truyền ngược (Backpropagation) (Slide 60 - 62)
- Biện pháp chống Quá khớp (Early Stopping) (Slide 31)
- Đánh giá bằng Ma trận nhầm lẫn (Confusion Matrix) (Slide 25)
"""

import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (64, 64)  # Resize nhỏ để trích đặc trưng pixel phẳng cho MLP


def extract_features_and_labels(split_name: str):
    """
    Trích xuất đặc trưng và nhãn từ thư mục ảnh.
    Pha trích chọn thuộc tính đặc trưng (Feature Extraction - Slide 17 & 18).
    """
    split_path = DATA_DIR / split_name
    X = []
    y = []

    print(f"[*] Đang trích xuất đặc trưng cho tập: {split_name}...")
    for idx, class_name in enumerate(CLASSES):
        class_folder = split_path / class_name
        if not class_folder.exists():
            continue
        for img_path in class_folder.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(img_path) as img:
                        img_resized = img.convert("RGB").resize(IMG_SIZE)
                        # Chuẩn hóa pixel về [0, 1]
                        feat = np.array(img_resized, dtype=np.float32).flatten() / 255.0
                        X.append(feat)
                        y.append(idx)
                except Exception as e:
                    print(f"Bỏ qua ảnh lỗi {img_path}: {e}")

    return np.array(X), np.array(y)


def train_and_evaluate_mlp():
    start_time = time.time()

    # 1. Tải và trích đặc trưng
    X_train, y_train = extract_features_and_labels("train")
    X_val, y_val = extract_features_and_labels("val")
    X_test, y_test = extract_features_and_labels("test")

    print(f"[✓] Đã trích xuất xong: Train={X_train.shape[0]} mẫu, Val={X_val.shape[0]} mẫu, Test={X_test.shape[0]} mẫu.")
    print(f"    Số chiều véc-tơ đặc trưng: {X_train.shape[1]} (64x64x3)")

    # 2. Khởi tạo Mạng Nơ-ron Nhiều Lớp (MLP)
    # Khớp đúng kiến trúc 2 lớp ẩn (Slide 59)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),  # 2 lớp ẩn
        activation="relu",               # Hàm kích hoạt phi tuyến
        solver="adam",                   # Tối ưu hóa lan truyền ngược
        alpha=0.001,                     # L2 regularization chống Overfitting
        batch_size=64,                   # Kích thước mini-batch
        learning_rate_init=0.001,        # Tốc độ học ban đầu
        max_iter=200,                    # Số vòng lặp tối đa
        early_stopping=True,             # Dừng sớm theo dõi Validation Loss (Slide 31)
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=42,
        verbose=True,
    )

    # 3. Huấn luyện mô hình (Training Phase - Slide 17)
    print("\n[*] Bắt đầu huấn luyện MLP với giải thuật Lan truyền ngược (Backpropagation)...")
    mlp.fit(X_train, y_train)

    train_duration = time.time() - start_time
    print(f"[✓] Huấn luyện hoàn tất trong {train_duration:.2f} giây.")

    # 4. Lưu mô hình đã học
    model_path = MODELS_DIR / "mlp_baseline.pkl"
    joblib.dump(mlp, model_path)
    print(f"[✓] Đã lưu mô hình tại: {model_path}")

    # 5. Đánh giá trên tập Kiểm tra (Test set) - Slide 24 & 25
    print("\n[*] Đánh giá hiệu năng trên Tập Kiểm thử (Test Set)...")
    y_pred = mlp.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"--> Độ chính xác (Accuracy) trên tập Test: {acc * 100:.2f}%\n")

    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=4)
    print("--- BẢNG ĐÁNH GIÁ CHI TIẾT (Classification Report) ---")
    print(report)

    # Lưu báo cáo text
    with open(REPORTS_DIR / "mlp_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"MLP Baseline Model Evaluation\n")
        f.write(f"Độ chính xác (Accuracy): {acc * 100:.2f}%\n")
        f.write(f"Thời gian huấn luyện: {train_duration:.2f}s\n\n")
        f.write(report)

    # 6. Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix - Slide 25)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Ma trận Nhầm lẫn (Confusion Matrix) — MLP Baseline", fontsize=14, fontweight="bold")
    plt.xlabel("Mô hình Dự đoán (Predicted)", fontsize=12)
    plt.ylabel("Thực tế (Ground Truth)", fontsize=12)
    plt.tight_layout()
    cm_path = REPORTS_DIR / "mlp_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu Ma trận nhầm lẫn tại: {cm_path}")

    # 7. Vẽ đường cong hàm lỗi (Loss Curve qua các vòng lặp)
    if hasattr(mlp, "loss_curve_"):
        plt.figure(figsize=(8, 5))
        plt.plot(mlp.loss_curve_, label="Training Loss", color="crimson", linewidth=2)
        plt.title("Đường cong Giảm Sai số (Loss Curve qua các Epochs) — MLP", fontsize=13)
        plt.xlabel("Vòng lặp (Iteration)", fontsize=11)
        plt.ylabel("Giá trị Lỗi (Loss)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        loss_path = REPORTS_DIR / "mlp_loss_curve.png"
        plt.savefig(loss_path, dpi=300)
        plt.close()
        print(f"[✓] Đã lưu Đường cong hàm lỗi tại: {loss_path}")

    print("\n[✓] Hoàn thành pipeline huấn luyện và đánh giá MLP Baseline!\n")


if __name__ == "__main__":
    train_and_evaluate_mlp()
