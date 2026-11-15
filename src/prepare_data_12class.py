"""
prepare_data_12class.py — Tự động tải và phân chia bộ dữ liệu 12 lớp rác thực tế
(Bao gồm cả Pin/Battery và Rác hữu cơ/Biological)
Phân chia: 70% Train, 15% Validation, 15% Test (Stratified Split theo chuẩn ml-data-auditor)
"""

import os
import shutil
import zipfile
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_URL = "https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes/resolve/main/garbage_classification.zip"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA12_DIR = BASE_DIR / "data" / "12class"

TRAIN_DIR = DATA12_DIR / "train"
VAL_DIR = DATA12_DIR / "val"
TEST_DIR = DATA12_DIR / "test"


def download_dataset():
    """Tải garbage_classification.zip (~239.5 MB) từ Hugging Face."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "garbage_classification_12class.zip"

    if zip_path.exists() and zip_path.stat().st_size > 200_000_000:
        print(f"[✓] Đã tìm thấy file nén: {zip_path}")
        return zip_path

    print(f"[*] Đang tải Bộ dữ liệu 12 Lớp (~239.5 MB) từ Hugging Face...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(DATASET_URL, stream=True, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Tải 12-Class Dataset"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[✓] Tải thành công: {zip_path}")
    return zip_path


def extract_and_split(zip_path):
    """Giải nén và phân chia Stratified Train/Val/Test."""
    extract_target = RAW_DIR / "extracted_12class"
    if not extract_target.exists():
        print(f"[*] Đang giải nén {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_target)
        print("[✓] Giải nén hoàn tất.")

    # Tìm thư mục chứa các thư mục con phân lớp
    source_root = extract_target
    if (extract_target / "garbage_classification").exists():
        source_root = extract_target / "garbage_classification"
    elif (extract_target / "Garbage classification").exists():
        source_root = extract_target / "Garbage classification"

    subfolders = [f for f in source_root.iterdir() if f.is_dir() and not f.name.startswith(".")]
    print(f"[*] Tìm thấy {len(subfolders)} lớp danh mục: {[f.name for f in subfolders]}")

    # Tạo thư mục đích
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

    total_train, total_val, total_test = 0, 0, 0

    print("[*] Đang phân chia dữ liệu Stratified (70% Train - 15% Val - 15% Test)...")
    for class_folder in subfolders:
        class_name = class_folder.name.lower().replace(" ", "_")
        images = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        if not images:
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

        for split, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            target_class_dir = DATA12_DIR / split / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for img in img_list:
                dst = target_class_dir / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)
        print(f"  - {class_name:15s}: {len(images):5d} ảnh (Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)})")

    print(f"\n[✓] Hoàn tất phân chia 12 lớp:")
    print(f"    Train: {total_train} ảnh (70%)")
    print(f"    Val:   {total_val} ảnh (15%)")
    print(f"    Test:  {total_test} ảnh (15%)")
    print(f"    Tổng:  {total_train + total_val + total_test} ảnh.")


if __name__ == "__main__":
    zip_file = download_dataset()
    extract_and_split(zip_file)
