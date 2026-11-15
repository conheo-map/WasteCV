# ♻️ WasteCV Station v2.0 — Hệ Thống Nhận Diện & Phân Loại Rác Thải Công Nghiệp (Production-Grade)

> **Đồ án môn:** Trí Tuệ Nhân Tạo — Khoa CNTT, Đại học Kinh Tế Quốc Dân (NEU)  
> **Kiến trúc SOTA kép:**  
> 1. 🎥 **YOLOv8 Multi-Object Detector:** Khoanh vùng đa vật thể rác di chuyển thời gian thực trên Camera (<10ms).  
> 2. 🔬 **12-Class SOTA Deep CNN (MobileNetV2):** Phân loại chuyên sâu đạt **96.01% Accuracy** trên 15.520 ảnh thực tế (có cả **Pin/Hazard** và **Rác Hữu Cơ/Biological**).  
> 3. 🧪 **6-Class TrashNet CNN (92.17%)** vs **MLP Baseline (46.74%)** phục vụ đối sánh học thuật Chương 6.  
> **Khung lý thuyết:** Bám sát 100% nội dung bài giảng Chương 6 — Học Máy (Machine Learning).

---

## 📌 1. Cấu Trúc Thư Mục Dự Án

```
C:\HocC\WasteCV\
├── data/
│   ├── raw/                 # File nén tải về từ HuggingFace
│   ├── train/, val/, test/  # Dữ liệu TrashNet 6 lớp (2.527 ảnh)
│   └── 12class/             # Dữ liệu công nghiệp 12 lớp (15.520 ảnh)
├── docs/
│   ├── BAO_CAO_DO_AN_WASTECV.md # Báo cáo đồ án toàn diện chuẩn học thuật
│   ├── SLIDE_THUYET_TRINH.md    # Kịch bản & 15 slides thuyết trình bảo vệ
│   └── BO_CAU_HOI_PHAN_BIEN.md  # 10 câu hỏi & đáp án phản biện trước Hội đồng
├── models/
│   ├── best_waste12_cnn.pt  # Mô hình SOTA 12 Lớp (Test Acc: 96.01% - 11.2 MB)
│   ├── best_cnn.pt          # Mô hình Deep CNN 6 Lớp (Test Acc: 92.17% - 9.8 MB)
│   ├── mlp_baseline.pkl     # Mô hình MLP Baseline 2 Lớp Ẩn (Test Acc: 46.74%)
│   └── yolov8n.pt           # Động cơ YOLOv8 Real-time Multi-Object Detector (6.2 MB)
├── reports/
│   ├── waste12_classification_report.txt # Báo cáo chi tiết 12 lớp (F1: 94.76%)
│   ├── waste12_confusion_matrix.png     # Ma trận nhầm lẫn 12x12
│   ├── waste12_learning_curves.png       # Đồ thị học tập 12 lớp
│   ├── cnn_confusion_matrix.png         # Ma trận nhầm lẫn 6x6
│   ├── model_comparison.png             # Biểu đồ đối sánh hiệu năng
│   └── comparison.md                    # Báo cáo phân tích chuyên sâu
├── src/
│   ├── detector.py              # Động cơ YOLOv8 nhận diện đa vật thể & cảnh báo nguy hại
│   ├── prepare_data_12class.py  # Tải & chia dữ liệu 12 lớp (15.520 ảnh)
│   ├── train_cnn_12class.py     # Huấn luyện SOTA 12-Class CNN (AMP + Cosine Annealing)
│   ├── prepare_data.py          # Tải & chia dữ liệu TrashNet 6 lớp
│   ├── train_cnn.py             # Huấn luyện MobileNetV2 6 lớp
│   ├── train_mlp.py             # Huấn luyện MLP Baseline
│   ├── evaluate.py              # Xuất báo cáo đối sánh các mô hình
│   └── app.py                   # Giao diện Web Station đa tab (Streamlit)
├── requirements.txt
└── README.md
```

---

## 📊 2. Bảng Đối Sánh Các Thế Hệ Mô Hình Đã Triển Khai

| Tiêu chí Đánh giá | Mạng Nơ-ron Nhiều Lớp (MLP) | MobileNetV2 (TrashNet 6 Lớp) | **SOTA Deep CNN (12 Lớp Thực Tế)** | **YOLOv8 Real-time Detector** |
| :--- | :---: | :---: | :---: | :---: |
| **Quy mô Dữ liệu** | 2.527 ảnh | 2.527 ảnh | **15.520 ảnh thực tế** | COCO + Multi-Stream |
| **Số Lớp Phân Loại** | 6 Lớp | 6 Lớp | **12 Lớp (có Pin & Rác Hữu Cơ)** | Đa vật thể đồng thời |
| **Độ chính xác (Accuracy)** | 46.74% | 92.17% | **96.01% (Test Set)** | mAP50 > 90% |
| **Macro F1-Score** | 41.08% | 90.67% | **94.76%** | Real-time Stream |
| **Độ nhạy Pin (`battery`)** | Không hỗ trợ | Không hỗ trợ | **100.00% (Recall tuyệt đối)** | Cảnh báo cháy nổ tức thì |
| **Độ nhạy Rác Hữu Cơ** | Không hỗ trợ | Không hỗ trợ | **98.65% (Recall)** | Phân luồng Compost |
| **Độ trễ suy luận** | ~3.2 ms | ~6.4 ms | **~7.1 ms** | **~8.5 ms (>100 FPS)** |
| **Đánh giá triển khai** | Baseline đối chứng | Đồ án học thuật xuất sắc | **Sẵn sàng thương mại hóa** | **Khoanh vùng đa rác camera** |

---

## 🚀 3. Hướng Dẫn Vận Hành Hệ Thống

### Bước 1: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện mô hình (Tất cả file trọng số đã train sẵn trong `models/`)
Nếu muốn tự huấn luyện lại từ đầu trên GPU RTX 2050:
```bash
# Huấn luyện mô hình SOTA 12 lớp (10.9 phút trên GPU CUDA)
python src/train_cnn_12class.py

# Hoặc huấn luyện mô hình 6 lớp TrashNet
python src/train_cnn.py
```

### Bước 3: Khởi chạy Trạm Phân Loại Rác Thông Minh (Web Station)
```bash
streamlit run src/app.py
```
👉 Mở trình duyệt tại: **`http://localhost:8501`**
- **Tab 1:** Quét đa vật thể rác bằng YOLOv8 thời gian thực (Webcam hoặc Upload).
- **Tab 2:** Phân loại chuyên sâu 12 lớp với phân bố xác suất Softmax.
- **Tab 3:** Bảng kiểm toán và đối sánh các thế hệ mô hình.

---

## 📚 4. Khung Lý Thuyết Bám Sát Chương 6 (NEU)
- **Học có giám sát (Supervised Learning - Slide 8, 14):** Bài toán phân loại rác đa lớp loại trừ lẫn nhau và phát hiện vật thể.
- **Bộ ba (T, P, E - Slide 11):** T=Nhận diện rác, P=Accuracy (96.01%)/F1 (94.76%), E=15.520 ảnh thực tế.
- **Mô hình Nơ-ron & Hàm kích hoạt (Slide 37, 38):** $Net = \sum w_j \cdot s_j$, ReLU & Softmax xác suất.
- **Lan truyền ngược & Tối ưu hóa (Slide 55, 60–62):** Gradient Descent với AdamW và Cosine Annealing.
- **Kiểm soát Quá khớp (Overfitting - Slide 26–31):** Khắc phục bằng Data Augmentation, Dropout và Early Stopping.
- **Ma trận nhầm lẫn (Confusion Matrix - Slide 25):** Đánh giá toàn diện 12x12 và 6x6.
