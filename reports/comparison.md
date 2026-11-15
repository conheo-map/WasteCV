# 📊 BÁO CÁO ĐỐI SÁNH HIỆU NĂNG CÁC MÔ HÌNH HỌC MÁY
> Môn: Trí Tuệ Nhân Tạo — Khoa CNTT, ĐH Kinh Tế Quốc Dân
> Đề tài: Hệ thống Nhận diện và Phân loại Rác thải bằng Thị giác Máy tính

---

## 1. Bảng Tổng Hợp Chỉ Số Đánh Giá (Test Set)

| Tiêu chí Đánh giá | Mạng Nơ-ron Nhiều Lớp (MLP Baseline) | Mạng Tích Chập Sâu (MobileNetV2 CNN) | Chênh lệch (Cải thiện) |
| :--- | :---: | :---: | :---: |
| **Kiến trúc mô hình** | 2 Lớp ẩn (256, 128 nơ-ron) | 53 Lớp (Inverted Residual Blocks) | +51 Lớp tích chập |
| **Phương thức trích chọn** | Thủ công (Flatten pixel 64x64) | Tự động qua các tầng Convolution | Vượt trội ngữ cảnh không gian |
| **Độ chính xác (Accuracy)** | **46.74%** | **92.17%** | **+45.43%** |
| **Độ chuẩn xác (Precision)** | **61.80%** | **89.67%** | **+27.87%** |
| **Độ bao phủ (Recall)** | **42.37%** | **92.72%** | **+50.35%** |
| **F1-Score (Macro Avg)** | **41.08%** | **90.67%** | **+49.59%** |
| **Độ trễ suy luận (Latency)** | ~3 ms / ảnh (CPU) | ~6 ms / ảnh (GPU RTX 2050) | Đạt chuẩn thời gian thực (<30ms) |

---

## 2. Nhận Xét & Phân Tích Chuyên Sâu (Đối chiếu Chương 6)

1. **Hiệu quả của Mạng tích chập sâu (Deep CNN) so với MLP cổ điển:**
   - Mạng MLP nhận đầu vào là vector 1 chiều phẳng nên bị mất hoàn toàn tương quan không gian 2D giữa các pixel (Spatial hierarchy).
   - Mô hình CNN tự động học được các bộ lọc trích xuất đường nét, cạnh, kết cấu vật liệu (vỏ lon kim loại sáng bóng, độ trong của chai thủy tinh/nhựa) ➔ Giúp tăng Accuracy lên vượt bậc (+45.4%).

2. **Khắc phục hiện tượng Quá khớp (Overfitting - Slide 28 & 31):**
   - Cả 2 mô hình đều được áp dụng cơ chế **Dừng sớm (Early Stopping)** khi loss trên tập Validation bắt đầu phân kỳ.
   - Mô hình MobileNetV2 kết hợp thêm kỹ thuật **Data Augmentation** (xoay, đổi màu, lật ảnh) giúp triệt tiêu hiện tượng học vẹt nền ảnh.
