"""
prepare_data.py — Tự động tải và phân chia dữ liệu TrashNet
Phân chia: 70% Train, 15% Validation, 15% Test (Stratified Split, seed=42)
Khớp đúng quy chuẩn Slide 24: Tập huấn luyện, Tập kiểm tra (validation), Tập đánh giá (test)
"""

import os
import shutil
import zipfile
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def download_dataset():
    """Tải dataset-resized.zip từ Hugging Face nếu chưa có."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "dataset-resized.zip"

    if zip_path.exists() and zip_path.stat().st_size > 10_000_000:
        print(f"[✓] Đã tìm thấy file nén: {zip_path}")
        return zip_path

    print(f"[*] Đang tải TrashNet dataset (~42.8 MB) từ Hugging Face...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(DATASET_URL, stream=True, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Tải TrashNet"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[✓] Tải thành công: {zip_path}")
    return zip_path


def extract_dataset(zip_path: Path):
    """Giải nén dataset."""
    extract_target = RAW_DIR / "dataset-resized"
    if extract_target.exists() and any(extract_target.iterdir()):
        print(f"[✓] Dữ liệu đã được giải nén tại: {extract_target}")
        return extract_target

    print(f"[*] Đang giải nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print(f"[✓] Giải nén hoàn tất!")
    return extract_target


def split_data(source_dir: Path):
    """
    Phân chia phân tầng (Stratified Split): 70% Train, 15% Val, 15% Test
    Tạo cấu trúc thư mục ImageFolder chuẩn PyTorch / YOLO
    """
    print(f"[*] Bắt đầu phân chia tập dữ liệu (70% Train / 15% Val / 15% Test)...")

    # Kiểm tra đường dẫn thư mục con
    classes_found = []
    for c in CLASSES:
        c_path = source_dir / c
        if not c_path.exists():
            # Thử tìm lồng nhau (dataset-resized/dataset-resized/...)
            alt_path = source_dir / "dataset-resized" / c
            if alt_path.exists():
                c_path = alt_path
        if c_path.exists():
            classes_found.append((c, c_path))

    if not classes_found:
        raise FileNotFoundError(f"Không tìm thấy các thư mục nhãn trong {source_dir}")

    # Xóa sạch thư mục train/val/test cũ nếu có để tránh trùng lặp
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for c in CLASSES:
            (split_dir / c).mkdir(exist_ok=True)

    summary = {c: {"train": 0, "val": 0, "test": 0, "total": 0} for c in CLASSES}

    for class_name, class_path in classes_found:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in class_path.iterdir() if f.suffix.lower() in valid_extensions]
        total = len(images)
        summary[class_name]["total"] = total

        # Tách Train (70%) và Phần còn lại (30%)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=42)
        # Tách Phần còn lại thành Val (15%) và Test (15%)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

        # Copy vào từng thư mục
        for img in train_imgs:
            shutil.copy2(img, TRAIN_DIR / class_name / img.name)
            summary[class_name]["train"] += 1

        for img in val_imgs:
            shutil.copy2(img, VAL_DIR / class_name / img.name)
            summary[class_name]["val"] += 1

        for img in test_imgs:
            shutil.copy2(img, TEST_DIR / class_name / img.name)
            summary[class_name]["test"] += 1

    # In bảng thống kê phân bố
    print("\n" + "=" * 65)
    print(f"{'Lớp rác (Class)':<15} | {'Train (70%)':<12} | {'Val (15%)':<12} | {'Test (15%)':<12} | {'Tổng':<8}")
    print("=" * 65)
    total_train = total_val = total_test = total_all = 0
    for c, counts in summary.items():
        print(f"{c:<15} | {counts['train']:<12} | {counts['val']:<12} | {counts['test']:<12} | {counts['total']:<8}")
        total_train += counts["train"]
        total_val += counts["val"]
        total_test += counts["test"]
        total_all += counts["total"]
    print("-" * 65)
    print(f"{'TỔNG CỘNG':<15} | {total_train:<12} | {total_val:<12} | {total_test:<12} | {total_all:<8}")
    print("=" * 65)
    print("[✓] Dữ liệu đã sẵn sàng cho huấn luyện tại thư mục data/ !\n")


if __name__ == "__main__":
    zip_p = download_dataset()
    src_p = extract_dataset(zip_p)
    split_data(src_p)
