# ♻️ WasteCV — Hệ Thống Nhận Diện & Phân Loại Rác Thải Thông Minh

> **Đồ án môn:** Trí Tuệ Nhân Tạo — Khoa CNTT, Đại học Kinh Tế Quốc Dân (NEU)  
> **Kiến trúc:** So sánh Mạng Nơ-ron Nhiều Lớp (MLP Baseline) vs Mạng Tích Chập Sâu (MobileNetV2 Deep CNN)  
> **Khung lý thuyết:** Bám sát 100% nội dung bài giảng Chương 6 — Học Máy (Machine Learning)

---

## 📌 1. Cấu Trúc Thư Mục Dự Án

```
C:\HocC\WasteCV\
├── data/
│   ├── raw/                 # File nén TrashNet tải về
│   ├── train/               # 70% dữ liệu (1.766 ảnh, 6 lớp)
│   ├── val/                 # 15% dữ liệu (378 ảnh, 6 lớp)
│   └── test/                # 15% dữ liệu (383 ảnh, 6 lớp)
├── docs/
│   ├── BAO_CAO_DO_AN_WASTECV.md # Báo cáo đồ án toàn diện chuẩn học thuật
│   ├── SLIDE_THUYET_TRINH.md    # Kịch bản & 15 slides thuyết trình bảo vệ
│   └── BO_CAU_HOI_PHAN_BIEN.md  # 10 câu hỏi & đáp án phản biện trước Hội đồng
├── models/
│   ├── mlp_baseline.pkl     # Trọng số Mạng Nơ-ron Nhiều Lớp (MLP)
│   └── best_cnn.pt          # Trọng số Deep CNN (MobileNetV2 Transfer Learning)
├── reports/
│   ├── mlp_confusion_matrix.png
│   ├── mlp_loss_curve.png
│   ├── cnn_learning_curves.png
│   ├── cnn_confusion_matrix.png
│   ├── model_comparison.png # Biểu đồ đối sánh hiệu năng
│   └── comparison.md        # Báo cáo phân tích chuyên sâu
├── src/
│   ├── prepare_data.py      # Tải & chia dữ liệu Stratified Split
│   ├── train_mlp.py         # Huấn luyện MLP Classifier (Backpropagation)
│   ├── train_cnn.py         # Huấn luyện MobileNetV2 trên GPU
│   ├── evaluate.py          # Xuất báo cáo so sánh các mô hình
│   └── app.py               # Giao diện Web Demo Streamlit
├── requirements.txt
└── README.md
```

---

## 🚀 2. Hướng Dẫn Chạy Hệ Thống

### Bước 1: Chuẩn bị môi trường
```bash
pip install -r requirements.txt
```

### Bước 2: Tải và chia dữ liệu (nếu chưa có)
```bash
python src/prepare_data.py
```

### Bước 3: Huấn luyện các mô hình
```bash
# Huấn luyện MLP Baseline (mất ~30 giây)
python src/train_mlp.py

# Huấn luyện MobileNetV2 Deep CNN trên GPU (mất ~3 phút)
python src/train_cnn.py
```

### Bước 4: Xuất báo cáo đối sánh
```bash
python src/evaluate.py
```

### Bước 5: Khởi chạy Giao diện Web Demo
```bash
streamlit run src/app.py
```
👉 Mở trình duyệt tại: `http://localhost:8501` để trải nghiệm tải ảnh hoặc bật Webcam phân loại rác thời gian thực!

---

## 📚 3. Đối Chiếu Kiến Thức Chương 6 (NEU)
- **Học có giám sát (Supervised Learning - Slide 8, 14):** Bài toán phân loại 6 lớp rác thải rời rạc.
- **Biểu diễn bài toán (T, P, E - Slide 11):** T=Phân loại rác, P=Accuracy/F1-score, E=Tập ảnh TrashNet.
- **Mô hình Nơ-ron & Hàm kích hoạt (Slide 37, 38):** $Net = \sum w_j \cdot s_j$, ReLU & Softmax.
- **Lan truyền ngược (Backpropagation - Slide 60–62):** Tối ưu hóa trọng số qua các Epochs.
- **Quá khớp (Overfitting - Slide 26–31):** Khắc phục bằng Data Augmentation, Dropout và Early Stopping.
- **Ma trận nhầm lẫn (Confusion Matrix - Slide 25):** Đánh giá chi tiết Precision, Recall, Accuracy.
