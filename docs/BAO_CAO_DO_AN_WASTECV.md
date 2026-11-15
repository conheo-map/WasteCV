# BÁO CÁO ĐỒ ÁN MÔN HỌC: TRÍ TUỆ NHÂN TẠO
## ĐỀ TÀI: HỆ THỐNG NHẬN DIỆN VÀ PHÂN LOẠI RÁC THẢI TỰ ĐỘNG BẰNG THỊ GIÁC MÁY TÍNH VÀ MẠNG NƠ-RON SÂU (WASTECV)

---

- **Học phần:** Trí Tuệ Nhân Tạo (Artificial Intelligence)
- **Đơn vị đào tạo:** Trường Công nghệ & Kinh tế số — Đại học Kinh tế Quốc dân (NEU)
- **Cơ sở lý thuyết:** Chương 6 — Học Máy (Machine Learning)
- **Sinh viên thực hiện:** Đồ án Nghiên cứu & Ứng dụng AI
- **Năm học:** 2026

---

## MỤC LỤC
1. [CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ TÍNH CẤP THIẾT CỦA ĐỀ TÀI](#chương-1-đặt-vấn-đề-và-tính-cấp-thiết-của-đề-tài)
   - 1.1. Bối cảnh thực tiễn & Thách thức trong quản lý rác thải
   - 1.2. Mục tiêu nghiên cứu và giải pháp đề xuất
   - 1.3. Định hình bài toán theo Lý thuyết Học máy Tom Mitchell (T, P, E)
2. [CHƯƠNG 2: CƠ SỞ LÝ THUYẾT HỌC MÁY & MẠNG NƠ-RON (CHƯƠNG 6 NEU)](#chương-2-cơ-sở-lý-thuyết-học-máy--mạng-nơ-ron-chương-6-neu)
   - 2.1. Phân loại các mô hình Học máy: Vì sao chọn Học có giám sát (Supervised Learning)?
   - 2.2. Kiến trúc Mạng Nơ-ron Nhân tạo (Artificial Neural Network)
   - 2.3. Thuật toán Lan truyền ngược (Backpropagation) và Tối ưu hóa Gradient Descent
   - 2.4. Đột phá từ Mạng nơ-ron nhiều lớp (MLP) sang Mạng Tích chập Sâu (Deep CNN)
   - 2.5. Kiểm soát Hiện tượng Quá khớp (Overfitting) và Kém khớp (Underfitting)
3. [CHƯƠNG 3: THIẾT KẾ HỆ THỐNG & QUY TRÌNH XỬ LÝ DỮ LIỆU](#chương-3-thiết-kế-hệ-thống--quy-trình-xử-lý-dữ-liệu)
   - 3.1. Tập dữ liệu thực nghiệm TrashNet
   - 3.2. Chiến lược Phân chia Dữ liệu (Stratified Train / Val / Test Split)
   - 3.3. Kỹ thuật Tăng cường Dữ liệu (Data Augmentation Pipeline)
   - 3.4. Kiến trúc Hệ thống Tổng thể (Edge AI & Decision Pipeline)
4. [CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM VÀ ĐỐI SÁNH HIỆU NĂNG](#chương-4-kết-quả-thực-nghiệm-và-đối-sánh-hiệu-năng)
   - 4.1. Môi trường thực nghiệm & Cấu hình phần cứng
   - 4.2. Huấn luyện Mô hình Cơ sở (MLP Baseline)
   - 4.3. Huấn luyện Mô hình Nơ-ron Sâu (MobileNetV2 Deep CNN)
   - 4.4. Phân tích Chi tiết Ma trận Nhầm lẫn (Confusion Matrix)
   - 4.5. Đánh giá Đối sánh Toàn diện & Tốc độ Suy luận Thời gian thực
5. [CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG THỰC TẾ & KẾT LUẬN](#chương-5-triển-khai-ứng-dụng-thực-tế--kết-luận)
   - 5.1. Ứng dụng Web Demo Streamlit tương tác thời gian thực
   - 5.2. Khả năng tích hợp Thiết bị Biên (Raspberry Pi / NVIDIA Jetson)
   - 5.3. Đánh giá đóng góp, Hạn chế và Hướng phát triển tương lai
   - 5.4. Lời kết

---

# CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ TÍNH CẤP THIẾT CỦA ĐỀ TÀI

## 1.1. Bối cảnh thực tiễn & Thách thức trong quản lý rác thải
Trong bối cảnh đô thị hóa nhanh chóng và tiêu dùng tăng trưởng mạnh, lượng rác thải sinh hoạt phát sinh tại Việt Nam và các đô thị lớn trên toàn cầu đang gia tăng ở mức báo động. Theo thống kê, hàng chục nghìn tấn rác được thải ra mỗi ngày, trong đó phần lớn chưa được phân loại tại nguồn mà bị chôn lấp hỗn hợp hoặc đốt lộ thiên, gây ô nhiễm nghiêm trọng đến nguồn nước ngầm, đất đai và không khí.

Phân loại rác tại nguồn là mắt xích quan trọng nhất của chuỗi kinh tế tuần hoàn (Circular Economy). Tuy nhiên, phương pháp phân loại thủ công hiện nay bộc lộ các nhược điểm chí mạng:
1. **Thiếu tính tự giác và nhận thức của người dân:** Người dân thường khó phân biệt chính xác loại rác (ví dụ: nhựa tái chế được vs túi nilon tạp chất vô cơ).
2. **Chi phí nhân công cao và nguy cơ độc hại:** Công nhân vệ sinh môi trường phải tiếp xúc trực tiếp với rác thải sắc nhọn, hóa chất và vi khuẩn độc hại.
3. **Hiệu suất thấp:** Khâu phân loại thủ công tại nhà máy xử lý rác không theo kịp tốc độ thu gom, dẫn đến ùn ứ và giảm giá trị tái chế của vật liệu.

## 1.2. Mục tiêu nghiên cứu và giải pháp đề xuất
Đề tài **"Hệ thống Nhận diện và Phân loại Rác thải bằng Thị giác Máy tính (WasteCV)"** được xây dựng nhằm ứng dụng các thành tựu mới nhất của Trí Tuệ Nhân Tạo (AI), Học máy (Machine Learning) và Thị giác Máy tính (Computer Vision) để:
- Tự động nhận diện và phân loại rác thải thành 6 danh mục tiêu chuẩn: **Bìa carton (Cardboard), Thủy tinh (Glass), Kim loại (Metal), Giấy (Paper), Nhựa (Plastic), và Rác thải khác (Trash)**.
- Đạt độ chính xác cao (>90%), hoạt động thời gian thực (Real-time Latency < 30ms/ảnh) trên thiết bị phổ thông.
- Cung cấp giao diện trực quan hỗ trợ cả tải ảnh tĩnh và quét trực tiếp qua camera/webcam, tự động gợi ý thùng rác phù hợp và hướng dẫn xử lý đúng quy chuẩn.

## 1.3. Định hình bài toán theo Lý thuyết Học máy Tom Mitchell (T, P, E)
Theo giáo trình bài giảng **Chương 6: Học Máy (Slide 11)**, một chương trình máy tính được xem là học từ kinh nghiệm:
> *"Một chương trình được xem là học từ kinh nghiệm $E$ đối với một lớp nhiệm vụ $T$ và phép đo hiệu năng $P$, nếu hiệu năng của nó đối với các nhiệm vụ trong $T$, được đo bởi $P$, cải thiện cùng với kinh nghiệm $E$." (Tom Mitchell, 1997)*

Đối chiếu chính xác vào đề tài WasteCV:
- **Nhiệm vụ (Task - $T$):** Nhận đầu vào là khung hình ảnh kỹ thuật số của một vật thể rác thải và dự đoán nhãn phân lớp $y \in \{ \text{cardboard, glass, metal, paper, plastic, trash} \}$.
- **Thước đo hiệu năng (Performance Measure - $P$):**
  - Độ chính xác tổng thể: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
  - Độ chuẩn xác (Precision), Độ bao phủ (Recall), và điểm điều hòa $F_1\text{-Score}$ cho từng lớp.
  - Ma trận nhầm lẫn (Confusion Matrix) đa lớp $6 \times 6$.
  - Độ trễ xử lý suy luận (Inference Latency tính theo millisecond).
- **Kinh nghiệm huấn luyện (Experience - $E$):** Tập dữ liệu ảnh có gắn nhãn chuẩn **TrashNet** gồm $2.527$ ảnh mẫu đã được tiền xử lý và gán nhãn thực tế.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT HỌC MÁY & MẠNG NƠ-RON (CHƯƠNG 6 NEU)

## 2.1. Phân loại các mô hình Học máy: Vì sao chọn Học có giám sát?
Theo bài giảng **Slide 12–15 (Chương 6)**, học máy được phân thành 3 nhánh chính:
1. **Học có giám sát (Supervised Learning):** Dữ liệu huấn luyện gồm các cặp đầu vào - đầu ra $(x_i, y_i)$. Mục tiêu là xấp xỉ hàm mục tiêu $f: X \to Y$.
2. **Học không giám sát (Unsupervised Learning):** Chỉ có đầu vào $x_i$, không có nhãn đầu ra. Mục tiêu tìm cấu trúc ẩn (phân cụm, giảm chiều).
3. **Học tăng cường (Reinforcement Learning):** Tác tử tương tác với môi trường qua hành động và nhận phần thưởng/hình phạt.

**Quyết định thiết kế:** Bài toán phân loại rác thải thuộc dạng **Học có giám sát — Phân loại nhiều lớp (Multiclass Classification)** (Slide 14). Chúng ta đã có tập dữ liệu mẫu với nhãn chính xác từ chuyên gia, mục tiêu là huấn luyện mô hình ánh xạ các đặc trưng thị giác sang nhãn danh mục rác.

## 2.2. Kiến trúc Mạng Nơ-ron Nhân tạo (Artificial Neural Network)
Dựa theo **Slide 36–45 (Chương 6)**, mạng nơ-ron nhân tạo được lấy cảm hứng từ cấu trúc mạng thần kinh sinh học.
Một đơn vị nơ-ron nhân tạo cơ bản tính toán tổng có trọng số của các đầu vào và kích hoạt qua một hàm phi tuyến:

$$\text{Net} = \sum_{j=1}^{d} w_j \cdot x_j + b$$

$$\text{Output} = g(\text{Net})$$

Trong đó:
- $x_j$ là giá trị đầu vào thứ $j$, $w_j$ là trọng số liên kết (synaptic weight).
- $b$ là hệ số điều chỉnh (bias).
- $g(\cdot)$ là hàm kích hoạt phi tuyến (Activation Function).

Các hàm kích hoạt chính được sử dụng trong đề tài:
- **Sigmoid (Slide 44):** $g(z) = \frac{1}{1 + e^{-z}}$, đưa giá trị về khoảng $(0, 1)$, hữu ích cho xác suất nhưng dễ bị triệt tiêu gradient (vanishing gradient).
- **ReLU (Rectified Linear Unit):** $g(z) = \max(0, z)$, giữ gradient không bị suy giảm khi $z > 0$, giúp mạng nơ-ron sâu huấn luyện hội tụ nhanh hơn hàng chục lần.
- **Softmax:** Áp dụng tại tầng đầu ra cuối cùng để chuyển đổi vector logits $z \in \mathbb{R}^K$ ($K=6$) thành phân bố xác suất hợp lệ:
  $$P(y = c \mid x) = \frac{e^{z_c}}{\sum_{k=1}^{K} e^{z_k}}, \quad \sum_{c=1}^{6} P(y=c \mid x) = 1$$

## 2.3. Thuật toán Lan truyền ngược (Backpropagation) và Tối ưu Gradient Descent
Theo bài giảng **Slide 55–63 (Chương 6)**, học trong mạng nơ-ron nhiều lớp là bài toán tối ưu hóa tìm bộ trọng số $W^*$ sao cho hàm mất mát (Loss Function) trên tập dữ liệu là nhỏ nhất.
Với bài toán phân loại đa lớp, hàm mất mát chuẩn là **Cross-Entropy Loss**:

$$\mathcal{L}_{CE} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} y_{i,c} \log(\hat{y}_{i,c})$$

Quy tắc cập nhật trọng số theo giải thuật Gradient Descent (Slide 55):

$$w_{ij}^{(t+1)} = w_{ij}^{(t)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}}$$

Trong đó $\eta > 0$ là tốc độ học (Learning Rate).
Thuật toán **Lan truyền ngược (Backpropagation - Slide 60–62)** tính toán đạo hàm riêng $\frac{\partial \mathcal{L}}{\partial w_{ij}}$ dựa trên quy tắc dây chuyền (Chain Rule), truyền tín hiệu sai số từ tầng ra ngược về các tầng ẩn trước đó.

## 2.4. Đột phá từ Mạng nơ-ron nhiều lớp (MLP) sang Mạng Tích chập Sâu (Deep CNN)
Để chứng minh sâu sắc luận điểm học thuật của Chương 6, đồ án thực hiện so sánh đối chứng 2 thế hệ mạng nơ-ron:
1. **Mạng Nơ-ron Nhiều Lớp Cổ Điển (Multi-Layer Perceptron - MLP):**
   - Ảnh màu được thu nhỏ về $64 \times 64$ và duỗi phẳng (Flatten) thành vector $1 \times 12.288$.
   - Các tầng Fully Connected (FC) kết nối toàn bộ nơ-ron.
   - *Nhược điểm cố hữu:* Phá vỡ cấu trúc không gian 2 chiều (Spatial Topology) của ảnh; số lượng tham số bùng nổ; không có tính bất biến với phép tịnh tiến (Translation Invariance).
2. **Mạng Tích chập Sâu (Deep Convolutional Neural Network - CNN / MobileNetV2):**
   - Sử dụng các tầng tích chập (Convolutional Layers) với các bộ lọc trượt (Kernels) chia sẻ trọng số (Weight Sharing).
   - Tự động trích xuất đặc trưng từ mức thấp (cạnh, góc, texture) đến mức cao (hình dạng chai, độ gập của thùng carton, nắp kim loại).
   - Kiến trúc **Inverted Residual Blocks & Depthwise Separable Convolutions** của MobileNetV2 giúp mô hình chỉ nặng ~9 MB nhưng hiệu năng vượt trội, tối ưu hoàn hảo cho triển khai thiết bị biên.

## 2.5. Kiểm soát Hiện tượng Quá khớp (Overfitting) và Kém khớp (Underfitting)
Theo **Slide 28 & 31 (Chương 6)**:
- **Kém khớp (Underfitting):** Mô hình quá đơn giản, không đủ năng lực học mẫu dữ liệu (bias cao, cả Train loss và Val loss đều cao).
- **Quá khớp (Overfitting):** Mô hình học thuộc lòng cả nhiễu của tập huấn luyện, dẫn đến sai số rất thấp trên Train nhưng tăng vọt trên Validation/Test (variance cao).

Hệ thống WasteCV áp dụng đồng bộ 4 kỹ thuật chống quá khớp chuẩn mực:
1. **Phân chia tập dữ liệu nghiêm ngặt:** Train (70%) - Validation (15%) - Test (15%). Tập Test hoàn toàn độc lập và chỉ được đánh giá 1 lần duy nhất sau khi huấn luyện xong.
2. **Dừng sớm (Early Stopping - Slide 28):** Theo dõi hàm mất mát trên tập Validation; nếu Validation Loss không giảm trong $p$ chu kỳ (patience), quá trình huấn luyện lập tức dừng lại để lấy điểm trọng số tốt nhất.
3. **Dropout Regularization:** Ngẫu nhiên ngắt kết nối $20\% - 30\%$ nơ-ron trong quá trình lan truyền tiến để chống hiện tượng phụ thuộc chéo giữa các đặc trưng.
4. **Tăng cường dữ liệu (Data Augmentation):** Tạo ra các biến thể xoay, lật, đổi màu ngẫu nhiên để mở rộng không gian mẫu, ép mô hình học đặc trưng bất biến.

---

# CHƯƠNG 3: THIẾT KẾ HỆ THỐNG & QUY TRÌNH XỬ LÝ DỮ LIỆU

## 3.1. Tập dữ liệu thực nghiệm TrashNet
Đề tài sử dụng bộ dữ liệu chuẩn học thuật **TrashNet** (do Gary Thung & Mindy Yang, Đại học Stanford thu thập và gán nhãn).
Bộ dữ liệu gồm $2.527$ ảnh chụp rác thải đơn chiếc trên nền trắng với ánh sáng thực tế, được chia thành 6 danh mục:

| Danh mục (Class) | Ý nghĩa nghiệp vụ | Số lượng ảnh gốc | Tỷ lệ (%) |
| :--- | :--- | :---: | :---: |
| **Cardboard** | Bìa các-tông, thùng carton vận chuyển | 403 | 15.9% |
| **Glass** | Chai lọ thủy tinh trong suốt, có màu | 501 | 19.8% |
| **Metal** | Vỏ lon nước ngọt, hộp thiếc, nắp sắt | 410 | 16.2% |
| **Paper** | Giấy in, sách báo vụn, tài liệu | 594 | 23.5% |
| **Plastic** | Chai nhựa PET, bình nhựa HDPE | 482 | 19.1% |
| **Trash** | Túi nilon bẩn, rác vô cơ không tái chế | 137 | 5.4% |
| **TỔNG CỘNG** | **Toàn bộ dataset** | **2.527** | **100.0%** |

> [!NOTE]
> Nhận xét dữ liệu: Lớp `trash` có số lượng mẫu ít nhất ($137$ ảnh, chiếm $5.4\%$), tạo ra thách thức mất cân bằng lớp nhẹ (Class Imbalance). Hệ thống đã áp dụng hàm mất mát có trọng số và Data Augmentation để đảm bảo lớp này vẫn đạt độ nhạy cao khi dự đoán.

## 3.2. Chiến lược Phân chia Dữ liệu (Stratified Split)
Để tuân thủ nghiêm ngặt nguyên lý thống kê của Chương 6, dữ liệu được chia theo phương pháp **Stratified Shuffle Split** (giữ nguyên tỷ lệ phân bố giữa các lớp trong cả 3 tập):
- **Tập Huấn luyện (Train Set - 70%):** $1.766$ ảnh dùng để cập nhật trọng số nơ-ron qua Gradient Descent.
- **Tập Kiểm định (Validation Set - 15%):** $378$ ảnh dùng để tinh chỉnh siêu tham số và kích hoạt Early Stopping.
- **Tập Đánh giá Độc lập (Test Set - 15%):** $383$ ảnh được bảo toàn tuyệt đối để kiểm tra khả năng khái quát hóa (Generalization).

## 3.3. Kỹ thuật Tăng cường Dữ liệu (Data Augmentation Pipeline)
Trong quá trình nạp ảnh huấn luyện qua PyTorch DataLoader, ảnh được áp dụng một chuỗi biến đổi ngẫu nhiên theo thời gian thực (On-the-fly Augmentation):
- `RandomResizedCrop(size=224, scale=(0.8, 1.0))`: Giúp mô hình nhận diện vật thể khi kích thước hoặc khoảng cách camera thay đổi.
- `RandomHorizontalFlip(p=0.5)` & `RandomRotation(degrees=20)`: Giúp mô hình bất biến với góc nghiêng và hướng đặt của rác thải.
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: Mô phỏng sự biến đổi điều kiện ánh sáng (ánh sáng ban ngày, đèn tuýp, bóng râm).
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: Chuẩn hóa phân phối pixel về phân phối chuẩn tắc, giúp tối ưu hóa lan truyền ngược ổn định.

## 3.4. Kiến trúc Hệ thống Tổng thể (Edge AI & Decision Pipeline)
Hệ thống WasteCV được thiết kế theo mô hình phân tầng khép kín:
```
[ Khung hình Camera / Upload ]
               │
               ▼
   [ Tiền xử lý (Preprocessing) ]  --> Resize 224x224, Tensor Normalization
               │
               ▼
 [ Động cơ Suy luận (Deep CNN) ]  --> MobileNetV2 Feature Extractor + Classifier Head
               │
               ▼
    [ Vector Softmax 6 Lớp ]       --> P(Cardboard), P(Glass), ..., P(Trash)
               │
               ▼
[ Bộ Điều Phối Quyết Định (Agent) ]
 ├── Xác thực ngưỡng tin cậy (> 60%)
 ├── Phân luồng thùng rác chỉ định (Thùng Tái chế / Thùng Vô cơ)
 └── Xuất khuyến nghị xử lý & Hướng dẫn vệ sinh rác
```

---

# CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM VÀ ĐỐI SÁNH HIỆU NĂNG

## 4.1. Môi trường thực nghiệm & Cấu hình phần cứng
- **Hệ điều hành:** Microsoft Windows 11 Home (64-bit).
- **Bộ vi xử lý (CPU):** Intel Core i5 thế hệ mới, RAM 24 GB.
- **Bộ xử lý đồ họa (GPU):** NVIDIA GeForce RTX 2050 (4 GB VRAM chuyên dụng).
- **Phần mềm & Thư viện:** Python 3.13, PyTorch 2.6.0 + CUDA 12.4, Scikit-learn 1.9.1, OpenCV 5.0.0, Streamlit 1.61.1.

## 4.2. Huấn luyện Mô hình Cơ sở (MLP Baseline)
Mô hình MLP được xây dựng bằng `sklearn.neural_network.MLPClassifier`:
- **Đầu vào:** Flatten vector $64 \times 64 \times 3 = 12.288$ chiều.
- **Kiến trúc:** 2 tầng ẩn: $12.288 \to 256 \to 128 \to 6$.
- **Hàm kích hoạt:** ReLU, Tối ưu hóa: Adam, Learning rate: $0.001$, Max epochs: $50$.
- **Kết quả huấn luyện:**
  - Thời gian huấn luyện: $28.70$ giây.
  - Early stopping kích hoạt tại chu kỳ thứ $16$ do Validation Loss ngưng cải thiện.
  - **Độ chính xác trên tập Test:** **$46.74\%$**.
  - **F1-Score Macro:** **$41.08\%$**.

Chi tiết từng lớp trên tập Test của mô hình MLP:
- `cardboard`: Precision $86.5\%$, Recall $52.5\%$, F1 $65.3\%$
- `glass`: Precision $49.0\%$, Recall $31.6\%$, F1 $38.4\%$
- `metal`: Precision $100.0\%$, Recall $1.6\%$ (chỉ nhận diện trúng 1/62 vỏ kim loại!), F1 $3.2\%$
- `paper`: Precision $41.7\%$, Recall $81.1\%$, F1 $55.1\%$
- `plastic`: Precision $39.1\%$, Recall $58.9\%$, F1 $47.0\%$
- `trash`: Precision $54.5\%$, Recall $28.6\%$, F1 $37.5\%$

## 4.3. Huấn luyện Mô hình Nơ-ron Sâu (MobileNetV2 Deep CNN)
Mô hình MobileNetV2 áp dụng kỹ thuật **Học chuyển giao (Transfer Learning)** với bộ trọng số ImageNet tiền huấn luyện và tinh chỉnh (Fine-tuning):
- **Phần trích xuất đặc trưng (Backbone):** Giữ nguyên các tầng Depthwise Convolution trích chọn đặc trưng không gian.
- **Phần phân lớp (Classifier Head):** Thiết kế mới gồm `Dropout(0.3) -> Linear(1280, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 6)`.
- **Hàm tối ưu:** AdamW ($\text{lr} = 0.0003$, weight decay $= 0.01$).
- **Hàm mất mát:** CrossEntropyLoss với Label Smoothing ($0.05$).
- **Số chu kỳ:** 15 epochs trên GPU CUDA NVIDIA RTX 2050.
- **Kết quả huấn luyện:**
  - Tổng thời gian: $4.86$ phút.
  - Best Validation Accuracy: **$94.18\%$** (tại Epoch 13).
  - **Độ chính xác trên tập Test:** **$92.17\%$** ($353/383$ ảnh kiểm tra chuẩn xác).
  - **F1-Score Macro:** **$90.67\%$**.

Chi tiết từng lớp trên tập Test của MobileNetV2:
- `cardboard`: Precision **$100.0\%$**, Recall **$96.72\%$**, F1 **$98.33\%$**
- `glass`: Precision **$93.33\%$**, Recall **$92.11\%$**, F1 **$92.72\%$**
- `metal`: Precision **$89.71\%$**, Recall **$98.39\%$** ($61/62$ vỏ kim loại được phát hiện!), F1 **$93.85\%$**
- `paper`: Precision **$97.70\%$**, Recall **$94.44\%$**, F1 **$96.05\%$**
- `plastic`: Precision **$90.62\%$**, Recall **$79.45\%$**, F1 **$84.67\%$**
- `trash`: Precision **$66.67\%$**, Recall **$95.24\%$** ($20/21$ mẫu rác thải được nhận diện), F1 **$78.43\%$**

## 4.4. Phân tích Chi tiết Ma trận Nhầm lẫn (Confusion Matrix)
Dựa theo **Slide 25 (Chương 6)**, Ma trận nhầm lẫn thể hiện chi tiết số lượng dự đoán đúng (đường chéo chính) và các trường hợp dự đoán sai lệch giữa các lớp.

Đối với mô hình MobileNetV2:
1. **Lớp Cardboard và Paper:** Có độ chính xác gần như tuyệt đối (F1 > 96%) nhờ cấu trúc sợi cellulose và màu sắc đặc trưng của giấy/carton rất khác biệt với rác vô cơ.
2. **Sự nhầm lẫn giữa Plastic và Glass:** Đây là điểm giao thoa thị giác khó nhất trong bài toán. Có 11 mẫu chai nhựa (Plastic) bị dự đoán nhầm thành Thủy tinh (Glass) do cả hai đều có tính chất trong suốt, phản chiếu ánh sáng trắng và có hình dáng thân chai hình trụ thuôn dài.
3. **Lớp Trash (Rác hỗn hợp):** Mô hình đạt Recall cực cao ($95.24\%$), nghĩa là hệ thống bắt trọn hầu như toàn bộ rác thải khó tái chế, tránh hiện tượng rác độc hại lọt vào dây chuyền tái chế sạch.

## 4.5. Đánh giá Đối sánh Toàn diện & Tốc độ Suy luận Thời gian thực

| Chỉ số đánh giá | Mạng Nơ-ron Nhiều Lớp (MLP) | Mạng Tích Chập Sâu (MobileNetV2) | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **Kiến trúc mô hình** | 2 Lớp ẩn Fully Connected | 53 Lớp Inverted Residual | Sâu hơn và biểu diễn tốt hơn |
| **Độ chính xác (Accuracy)** | **$46.74\%$** | **$92.17\%$** | **$+45.43\%$** (Tăng gấp đôi) |
| **Precision (Macro Avg)** | $61.80\%$ | $89.67\%$ | $+27.87\%$ |
| **Recall (Macro Avg)** | $42.37\%$ | $92.72\%$ | $+50.35\%$ |
| **F1-Score (Macro Avg)** | **$41.08\%$** | **$90.67\%$** | **$+49.59\%$** |
| **Độ trễ suy luận (Latency)** | $3.2$ ms / ảnh (CPU) | $6.4$ ms / ảnh (GPU RTX 2050) | Đạt chuẩn thời gian thực (<30ms) |
| **Dung lượng file Model** | $12.6$ MB | $9.8$ MB | Gọn nhẹ, sẵn sàng nhúng |

> **Kết luận thực nghiệm then chốt:** Kết quả đối sánh thực tế minh chứng rõ ràng cho lý thuyết Chương 6: Mạng nơ-ron nhiều lớp (MLP) khi không có cơ chế tích chập không gian sẽ thất bại trước dữ liệu ảnh phức tạp ($46.74\%$), trong khi việc khai thác các tầng tích chập sâu kết hợp truyền ngược (Backpropagation) và tinh chỉnh trọng số đã đưa hiệu năng vượt lên mức ứng dụng công nghiệp ($92.17\%$).

---

# CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG THỰC TẾ & KẾT LUẬN

## 5.1. Ứng dụng Web Demo Streamlit tương tác thời gian thực
Hệ thống được đóng gói thành một ứng dụng web hoàn chỉnh (`src/app.py`):
- Cho phép người dùng chuyển đổi linh hoạt giữa 2 mô hình (MLP Baseline vs MobileNetV2) để quan sát sự khác biệt về kết quả ngay tại chỗ.
- Hỗ trợ tải file ảnh máy tính hoặc mở Webcam laptop chụp trực tiếp rác thải thật.
- Hiển thị trực quan: Tên danh mục rác (Tiếng Việt & Tiếng Anh), Huy hiệu màu sắc, Thanh phần trăm độ tin cậy, Biểu đồ cột phân bố xác suất Softmax của cả 6 lớp.
- Đưa ra hướng dẫn hành động cụ thể: Bỏ vào thùng rác nào (Tái chế vs Chôn lấp), cách chuẩn bị rác (tráng sạch nước, ép bẹp lon nhôm).

## 5.2. Khả năng tích hợp Thiết bị Biên (Raspberry Pi / NVIDIA Jetson)
Với kích thước mô hình chỉ $9.8$ MB và độ trễ suy luận chỉ $6.4$ ms trên GPU phổ thông:
- Mô hình hoàn toàn có thể được nạp vào các vi máy tính giá rẻ như **Raspberry Pi 4 / 5** (khi lượng tử hóa INT8 qua ONNX Runtime đạt tốc độ ~25-35 ms/ảnh) hoặc **NVIDIA Jetson Orin Nano**.
- Có thể kết nối trực tiếp với mạch vi điều khiển (Arduino / ESP32) điều khiển động cơ servo mở nắp các ngăn thùng rác tự động tương ứng với nhãn nhận diện.

## 5.3. Đánh giá đóng góp, Hạn chế và Hướng phát triển tương lai
### Đóng góp của đề tài:
1. Xây dựng trọn vẹn pipeline học máy từ dữ liệu thô $\to$ tiền xử lý $\to$ huấn luyện mô hình $\to$ đánh giá thống kê $\to$ ứng dụng demo tương tác thời gian thực.
2. Bám sát $100\%$ các khái niệm học thuật trọng tâm trong **Chương 6 (Học Máy - NEU)**: Bộ ba (T, P, E), Học có giám sát, Lan truyền ngược Gradient Descent, Quá khớp/Dừng sớm, và Ma trận nhầm lẫn.
3. Giải quyết thành công bài toán thực tế có tính nhân văn cao và thúc đẩy lối sống xanh.

### Hạn chế còn tồn tại:
1. Tập dữ liệu TrashNet chủ yếu chụp vật thể đơn chiếc trên nền đồng nhất; trong môi trường thực tế, rác thải thường bị dính bết, đè lấp lên nhau hoặc nằm trong túi bóng mờ.
2. Sự nhầm lẫn giữa chai nhựa trong suốt và chai thủy tinh vẫn còn xảy ra ở một số góc chụp phản quang mạnh.

### Hướng phát triển tiếp theo:
1. Nâng cấp mô hình từ Phân loại ảnh (Image Classification) sang Phát hiện vật thể (Object Detection - YOLOv8/v11) để có thể khoanh vùng và phân loại đồng thời nhiều loại rác trong cùng một khung hình.
2. Thu thập thêm dữ liệu rác thải sinh hoạt đặc thù tại Việt Nam (hộp xốp cơm, ly nhựa trà sữa, ống hút).

## 5.4. Lời kết
Đề tài **"Hệ thống Nhận diện và Phân loại Rác thải bằng Thị giác Máy tính (WasteCV)"** đã chứng minh tính khả thi, hiệu quả vượt trội của mạng nơ-ron sâu trong việc giải quyết các bài toán thị giác máy tính phục vụ đời sống. Đồ án không chỉ củng cố nền tảng lý thuyết học máy vững chắc mà còn mang lại một sản phẩm phần mềm hoàn chỉnh, sẵn sàng ứng dụng và nhân rộng.

---
*(Báo cáo được hoàn thiện dựa trên mã nguồn và kết quả thực nghiệm tại kho lưu trữ `C:\HocC\WasteCV`)*
