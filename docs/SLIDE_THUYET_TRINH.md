# KỊCH BẢN & CẤU TRÚC SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN AI
## ĐỀ TÀI: WASTECV — HỆ THỐNG NHẬN DIỆN VÀ PHÂN LOẠI RÁC THẢI BẰNG THỊ GIÁC MÁY TÍNH
> Môn học: Trí Tuệ Nhân Tạo — Đại học Kinh Tế Quốc Dân (NEU)  
> Thời lượng trình bày lý tưởng: 10 – 12 phút (kèm 2 phút Live Demo)

---

## TỔNG QUAN CẤU TRÚC BỘ SLIDE (15 SLIDES)

```
[Slide 1] Trang bìa & Giới thiệu đề tài
[Slide 2] Bối cảnh thực tế & Tính cấp thiết
[Slide 3] Định hình bài toán theo Lý thuyết Tom Mitchell (T, P, E)
[Slide 4] Kiến trúc Luồng Dữ liệu Đầu cuối (Data Pipeline)
[Slide 5] Cơ sở Lý thuyết: Học có giám sát & Mạng Nơ-ron Nhân tạo
[Slide 6] Thuật toán Tối ưu: Lan truyền ngược (Backpropagation) & Gradient Descent
[Slide 7] Tập dữ liệu thực nghiệm TrashNet & Chiến lược Phân chia
[Slide 8] Kỹ thuật Kiểm soát Quá khớp (Overfitting) & Data Augmentation
[Slide 9] Mô hình Cơ sở: Mạng Nơ-ron Nhiều Lớp (MLP Baseline)
[Slide 10] Mô hình Nâng cao: Mạng Tích Chập Sâu (MobileNetV2 Deep CNN)
[Slide 11] Đối sánh Hiệu năng: MLP vs MobileNetV2 (+45.4% Accuracy)
[Slide 12] Phân tích Chi tiết Ma trận Nhầm lẫn (Confusion Matrix)
[Slide 13] Ứng dụng Web Demo Streamlit Thời gian thực
[Slide 14] Đánh giá Khả năng Triển khai Thiết bị Biên (Edge AI) & Hạn chế
[Slide 15] Kết luận & Hướng phát triển tương lai
```

---

### SLIDE 1: TRANG BÌA & GIỚI THIỆU ĐỀ TÀI
- **Tiêu đề lớn:** HỆ THỐNG NHẬN DIỆN VÀ PHÂN LOẠI RÁC THẢI BẰNG THỊ GIÁC MÁY TÍNH (WASTECV)
- **Tiêu đề phụ:** Ứng dụng Mạng Nơ-ron Sâu & Học Chuyển Giao Phục Vụ Kinh Tế Tuần Hoàn
- **Thông tin học phần:** Bộ môn Trí Tuệ Nhân Tạo — Khoa CNTT & KT Số, NEU
- **Người thực hiện:** Nhóm nghiên cứu WasteCV
- **Visual:** Logo NEU, hình ảnh minh họa thùng rác thông minh kết hợp mắt camera AI.
- **Lời thoại thuyết trình (Speaker Script):**
  > *"Kính thưa Thầy Cô và các bạn! Hôm nay, em xin đại diện nhóm trình bày đồ án môn Trí Tuệ Nhân Tạo với đề tài: 'Hệ thống nhận diện và phân loại rác thải tự động bằng Thị giác máy tính - WasteCV'. Đề tài của chúng em được xây dựng bám sát kiến thức nền tảng của Chương 6: Học máy, nhằm giải quyết một vấn đề nhức nhối trong thực tiễn đô thị hiện nay."*

---

### SLIDE 2: BỐI CẢNH THỰC TẾ & TÍNH CẤP THIẾT
- **Nội dung chính:**
  - Lượng rác thải sinh hoạt phát sinh hàng chục ngàn tấn mỗi ngày; tỷ lệ chôn lấp hỗn hợp >70%.
  - Phân loại thủ công: Chi phí cao, nguy cơ bệnh nghề nghiệp, hiệu suất chậm.
  - Phân loại rác tại nguồn là điều kiện tiên quyết của tái chế và kinh tế tuần hoàn.
- **Mục tiêu của đề tài:**
  - Tự động hóa phân loại 6 dòng rác thải chính: Carton, Thủy tinh, Kim loại, Giấy, Nhựa, Rác khác.
  - Đạt độ chính xác cao (>90%) và độ trễ siêu thấp (<30ms) để lắp đặt vào các trạm thùng rác thông minh.
- **Lời thoại thuyết trình:**
  > *"Như Thầy Cô đã biết, nút thắt lớn nhất của ngành tái chế không phải là công nghệ xử lý tại nhà máy, mà là khâu phân loại rác tại nguồn. Nếu rác thải được phân loại chính xác ngay từ khoảnh khắc người dân vứt rác, giá trị tài nguyên tái chế sẽ tăng vọt. Xuất phát từ nhu cầu đó, nhóm đã nghiên cứu và phát triển giải pháp WasteCV — một trợ lý thị giác máy tính nhận diện và hướng dẫn phân loại rác tự động."*

---

### SLIDE 3: ĐỊNH HÌNH BÀI TOÁN THEO LÝ THUYẾT TOM MITCHELL (T, P, E)
- **Đối chiếu học thuật (Slide 11 Chương 6):**
  - **Nhiệm vụ (Task - $T$):** Tiếp nhận ảnh đầu vào từ camera và phân loại chính xác vào 1 trong 6 danh mục rác: $\{ \text{Cardboard, Glass, Metal, Paper, Plastic, Trash} \}$.
  - **Thước đo hiệu năng (Performance - $P$):** Accuracy, Macro Precision, Recall, F1-Score, Ma trận nhầm lẫn (Confusion Matrix), và Tốc độ xử lý (Inference Latency tính bằng ms).
  - **Kinh nghiệm (Experience - $E$):** Tập dữ liệu $2.527$ ảnh mẫu TrashNet có gán nhãn thực tế được nạp vào mạng nơ-ron để học các đặc trưng thị giác.
- **Hình ảnh gợi ý:** Sơ đồ tương tác bộ ba (T, P, E) theo giáo trình Tom Mitchell.
- **Lời thoại thuyết trình:**
  > *"Để đảm bảo tính chuẩn mực về khoa học máy tính, chúng em định nghĩa bài toán theo đúng mô hình của Tom Mitchell trong bài giảng Chương 6: Nhiệm vụ T là phân loại ảnh 6 lớp; Thước đo P là Accuracy, F1-Score và Confusion Matrix; còn Kinh nghiệm E chính là tập dữ liệu huấn luyện hơn 2.500 ảnh thực tế mà mô hình được học."*

---

### SLIDE 4: KIẾN TRÚC LUỒNG DỮ LIỆU ĐẦU CUỐI (DATA PIPELINE)
- **Quy trình 5 bước:**
  1. **Image Acquisition:** Nhận luồng ảnh từ Camera Webcam hoặc File Upload.
  2. **Preprocessing:** Resize ảnh chuẩn, chuẩn hóa kênh màu RGB theo mean/std của ImageNet.
  3. **Deep CNN Inference:** Trích xuất đặc trưng sâu qua MobileNetV2 Backbone và tính xác suất tại Classifier Head.
  4. **Post-processing & Thresholding:** Lọc ngưỡng độ tin cậy ($>60\%$) để loại bỏ phán đoán không chắc chắn.
  5. **Orchestrator Decision:** Ánh xạ nhãn ra chỉ dẫn thùng rác (Thùng tái chế / Thùng vô cơ) và xuất hướng dẫn vệ sinh rác.
- **Hình ảnh gợi ý:** Sơ đồ khối kiến trúc hệ thống (Flowchart).
- **Lời thoại thuyết trình:**
  > *"Kiến trúc của WasteCV là một luồng dữ liệu khép kín: từ lúc camera thu nhận ảnh, đi qua tiền xử lý, nạp vào mạng nơ-ron sâu để trích xuất đặc trưng, tính toán phân bố xác suất Softmax, và cuối cùng bộ điều phối sẽ chỉ định chính xác người dùng nên bỏ vật dụng đó vào thùng rác nào."*

---

### SLIDE 5: CƠ SỞ LÝ THUYẾT — HỌC CÓ GIÁM SÁT & MẠNG NƠ-RON
- **Điểm nhấn lý thuyết (Slide 14 & 36–45 Chương 6):**
  - **Dạng bài toán:** Học có giám sát (Supervised Learning) — Phân lớp đa lớp (Multiclass Classification).
  - **Mô hình toán học nơ-ron:**
    $$\text{Net} = \sum_{j=1}^{d} w_j \cdot x_j + b, \quad \text{Output} = g(\text{Net})$$
  - **Hàm kích hoạt (Activation):** ReLU ở các lớp ẩn để chống suy giảm đạo hàm; Softmax ở lớp đầu ra để chuyển thành xác suất tổng bằng 1.
- **Lời thoại thuyết trình:**
  > *"Về cơ sở lý thuyết, bài toán thuộc nhóm Học có giám sát. Mỗi nơ-ron tính tổng có trọng số các tín hiệu đầu vào kết hợp hàm kích hoạt phi tuyến ReLU. Tại lớp ra cuối cùng, hàm Softmax được sử dụng để đưa đầu ra về một phân phối xác suất hợp lệ trên 6 lớp rác thải."*

---

### SLIDE 6: THUẬT TOÁN TỐI ƯU — LAN TRUYỀN NGƯỢC (BACKPROPAGATION)
- **Điểm nhấn lý thuyết (Slide 55–63 Chương 6):**
  - **Hàm mất mát:** Cross-Entropy Loss: $\mathcal{L}_{CE} = - \sum_{c=1}^{6} y_c \log(\hat{y}_c)$
  - **Lan truyền tiến (Forward Pass):** Tính toán dự đoán và xác định sai số mất mát giữa nhãn thực tế $y$ và dự đoán $\hat{y}$.
  - **Lan truyền ngược (Backward Pass):** Áp dụng quy tắc dây chuyền (Chain Rule) để tính gradient đạo hàm riêng $\frac{\partial \mathcal{L}}{\partial w}$.
  - **Tối ưu hóa trọng số:** Thuật toán AdamW tự thích ứng tốc độ học (Adaptive Learning Rate), cập nhật: $w \leftarrow w - \eta \cdot \nabla \mathcal{L}$.
- **Lời thoại thuyết trình:**
  > *"Trọng tim của quá trình học là giải thuật Lan truyền ngược kết hợp tối ưu hóa Gradient Descent theo Slide 60. Sai số từ hàm Cross-Entropy được truyền ngược từ lớp ra về các lớp trước, đạo hàm riêng được tính toán để cập nhật bộ trọng số sao cho sai số dự đoán giảm dần sau mỗi chu kỳ huấn luyện."*

---

### SLIDE 7: TẬP DỮ LIỆU TRASHNET & CHIẾN LƯỢC PHÂN CHIA
- **Đặc điểm dữ liệu:**
  - $2.527$ ảnh chia thành 6 lớp: Cardboard ($403$), Glass ($501$), Metal ($410$), Paper ($594$), Plastic ($482$), Trash ($137$).
  - Có hiện tượng mất cân bằng lớp nhẹ ở lớp `trash` ($5.4\%$).
- **Phân chia Stratified Split (Giữ nguyên phân bố):**
  - **Train (70%):** $1.766$ ảnh để huấn luyện cập nhật trọng số.
  - **Validation (15%):** $378$ ảnh để điều chỉnh siêu tham số và Early Stopping.
  - **Test (15%):** $383$ ảnh độc lập hoàn toàn để đánh giá năng lực khái quát hóa.
- **Lời thoại thuyết trình:**
  > *"Chúng em sử dụng bộ dữ liệu TrashNet của Stanford. Để đánh giá một cách trung thực nhất, dữ liệu được chia theo tỷ lệ Stratified 70-15-15, đảm bảo tập Test 383 ảnh hoàn toàn độc lập và phản ánh khách quan năng lực nhận diện của mô hình khi gặp dữ liệu chưa từng thấy."*

---

### SLIDE 8: KIỂM SOÁT QUÁ KHỚP (OVERFITTING) & DATA AUGMENTATION
- **Lý thuyết Slide 28 & 31:** Phân biệt Underfitting vs Overfitting; nguy cơ mô hình 'học vẹt' nền ảnh.
- **4 Giải pháp triệt tiêu Overfitting trong WasteCV:**
  1. **Data Augmentation:** Tự động xoay $\pm 20^\circ$, lật ngang ngẫu nhiên, thay đổi độ sáng và tương phản (ColorJitter).
  2. **Dropout (0.3 và 0.2):** Ngắt kết nối ngẫu nhiên một phần nơ-ron để tránh phụ thuộc đặc trưng.
  3. **Early Stopping:** Dừng train ngay khi Validation Loss có xu hướng tăng trở lại.
  4. **Label Smoothing (0.05):** Giảm bớt sự tự tin thái quá của hàm Softmax.
- **Lời thoại thuyết trình:**
  > *"Để ngăn chặn hiện tượng Quá khớp (Overfitting) được nhấn mạnh trong Slide 31, nhóm áp dụng kỹ thuật Tăng cường dữ liệu (Data Augmentation) làm giàu tập ảnh, kết hợp kỹ thuật Dừng sớm (Early Stopping) và Dropout nhằm đảm bảo mô hình không học vẹt mà thực sự học được bản chất của vật thể."*

---

### SLIDE 9: MÔ HÌNH CƠ SỞ — MẠNG NƠ-RON NHIỀU LỚP (MLP BASELINE)
- **Kiến trúc mô hình:**
  - Đầu vào: Vector phẳng $64 \times 64 \times 3 = 12.288$ chiều.
  - 2 tầng ẩn: $12.288 \to 256 \to 128 \to 6$.
- **Kết quả thực nghiệm:**
  - Thời gian huấn luyện: $28.7$ giây.
  - Early Stopping dừng ở vòng lặp 16.
  - **Accuracy Test Set: 46.74% | F1-Score: 41.08%**.
- **Điểm yếu chí mạng:**
  - Lớp Kim loại (Metal) chỉ đạt Recall $1.61\%$ (chỉ đoán đúng 1 trên 62 vỏ lon).
  - Mất hoàn toàn cấu trúc không gian 2D và tương quan giữa các điểm ảnh lân cận.
- **Visual:** Chèn ảnh `reports/mlp_loss_curve.png` và `reports/mlp_confusion_matrix.png`.
- **Lời thoại thuyết trình:**
  > *"Trước tiên, nhóm xây dựng mô hình cơ sở là Mạng nơ-ron nhiều lớp (MLP). Kết quả trên tập Test chỉ đạt 46.74%. Đáng chú ý, lớp vỏ lon kim loại có độ nhạy chỉ 1.6%, mô hình gần như 'mù' trước kim loại. Lý do là vì khi duỗi phẳng ảnh thành vector 1D, các mối liên hệ không gian giữa các pixel liền kề bị phá vỡ hoàn toàn."*

---

### SLIDE 10: MÔ HÌNH NÂNG CAO — MẠNG TÍCH CHẬP SÂU (MOBILENETV2 CNN)
- **Kiến trúc & Tối ưu:**
  - Áp dụng kỹ thuật Học chuyển giao (Transfer Learning) với 53 tầng tích chập sâu.
  - Cấu trúc Inverted Residual & Depthwise Separable Convolutions giúp giảm $80\%$ số lượng phép tính so với CNN tiêu chuẩn.
- **Kết quả huấn luyện:**
  - Thời gian: 4.86 phút (15 epochs) trên GPU NVIDIA RTX 2050.
  - **Accuracy Test Set: 92.17% | F1-Score Macro: 90.67%**.
  - Lớp Metal tăng từ $1.61\%$ lên **$98.39\%$ Recall** ($61/62$ vỏ lon kim loại được nhận diện chính xác).
  - Lớp Cardboard đạt độ chuẩn xác Precision tuyệt đối: **$100.0\%$**.
- **Visual:** Chèn ảnh `reports/cnn_learning_curves.png`.
- **Lời thoại thuyết trình:**
  > *"Để khắc phục triệt để nhược điểm của MLP, nhóm nâng cấp lên Mạng Tích chập Sâu MobileNetV2. Nhờ các bộ lọc tích chập tự động trích xuất cạnh, vân và hình khối vật thể, mô hình đạt độ chính xác ấn tượng 92.17% trên tập kiểm tra độc lập, trong đó độ nhạy với vỏ kim loại nhảy vọt lên 98.4%."*

---

### SLIDE 11: ĐỐI SÁNH HIỆU NĂNG — MLP VS MOBILENETV2
- **Bảng đối sánh tổng hợp:**

| Tiêu chí | Mạng Nơ-ron Nhiều Lớp (MLP) | Mạng Tích Chập Sâu (MobileNetV2) | Chênh lệch (Cải thiện) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác (Accuracy)** | **46.74%** | **92.17%** | **+45.43%** (Gấp đôi) |
| **Precision (Macro Avg)** | 61.80% | 89.67% | +27.87% |
| **Recall (Macro Avg)** | 42.37% | 92.72% | +50.35% |
| **F1-Score (Macro Avg)** | 41.08% | 90.67% | +49.59% |
| **Tốc độ suy luận (Latency)** | 3.2 ms / ảnh | 6.4 ms / ảnh | Đáp ứng chuẩn thời gian thực |

- **Visual:** Chèn biểu đồ cột đối sánh `reports/model_comparison.png`.
- **Lời thoại thuyết trình:**
  > *"Trên slide là biểu đồ đối sánh trực quan giữa hai mô hình. MobileNetV2 vượt trội hơn MLP ở mọi chỉ số, đặc biệt F1-score tăng gần 50 điểm phần trăm. Quan trọng hơn, độ trễ suy luận của MobileNetV2 chỉ 6.4 ms/khung hình trên GPU, hoàn toàn đáp ứng tiêu chuẩn thời gian thực 30 khung hình/giây của hệ thống camera công nghiệp."*

---

### SLIDE 12: PHÂN TÍCH MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
- **Điểm nhấn lý thuyết (Slide 25 Chương 6):**
  - Trục tung là Nhãn thực tế (True Label), Trục hoành là Nhãn dự đoán (Predicted Label).
  - Đường chéo chính thể hiện các trường hợp nhận diện đúng.
- **Phát hiện thú vị từ thực nghiệm:**
  - `Cardboard` và `Paper` nhận diện cực kỳ chuẩn xác (F1 đạt $98\%$ và $96\%$).
  - Có 11 mẫu `Plastic` bị nhầm sang `Glass` ($15\%$): Do cả 2 đều là vật thể hình trụ tròn, trong suốt và có tính phản quang tương đồng dưới ánh sáng trắng.
  - Lớp `Trash` đạt Recall $95.2\%$ ($20/21$ mẫu bắt trúng), đảm bảo rác bẩn không bị lọt vào dây chuyền tái chế sạch.
- **Visual:** Chèn ảnh `reports/cnn_confusion_matrix.png`.
- **Lời thoại thuyết trình:**
  > *"Dựa vào Ma trận nhầm lẫn ở Slide 25 của thầy, chúng em phân tích sâu hơn: Đường chéo chính rất đậm màu, chứng tỏ mô hình dự đoán chính xác đa số mẫu. Điểm nhầm lẫn dễ hiểu nhất là giữa chai nhựa và chai thủy tinh do cùng có độ trong suốt và phản quang. Tuy nhiên, lớp rác bẩn 'Trash' có độ nhạy lên tới 95.2%, giúp hệ thống lọc sạch rác không thể tái chế."*

---

### SLIDE 13: ỨNG DỤNG WEB DEMO STREAMLIT THỜI GIAN THỰC
- **Tính năng nổi bật của Demo:**
  - Giao diện trực quan, đa ngôn ngữ, thiết kế responsive.
  - Cho phép người dùng chuyển đổi trực tiếp giữa 2 mô hình (MLP vs CNN) để thấy sự khác biệt.
  - 2 chế độ đầu vào: Tải file ảnh hoặc Mở camera webcam trực tiếp.
  - Hiển thị xác suất Softmax 6 lớp bằng biểu đồ trực quan.
  - Đưa ra hướng dẫn phân loại: Bỏ vào thùng rác nào và lưu ý tái chế (ép bẹp lon, rửa sạch chai).
- **Visual:** Ảnh chụp màn hình giao diện Web Streamlit đang nhận diện một chai nhựa / vỏ lon lon bia.
- **Lời thoại thuyết trình:**
  > *"Bên cạnh mô hình toán học, nhóm đã đóng gói thành công ứng dụng Web tương tác thời gian thực bằng Streamlit. Người dùng có thể bật webcam, giơ một vỏ lon nước ngọt hoặc hộp giấy trước camera, hệ thống sẽ ngay lập tức trả về nhãn nhận diện, độ tin cậy và hướng dẫn thùng rác chỉ định trong chưa đầy 10 mili giây. Sau đây em xin phép được demo trực tiếp 1 phút trên màn hình."*

---

### SLIDE 14: ĐỘT PHÁ CÔNG NGHỆ — WASTECV STATION V2.0 (PRODUCTION-GRADE)
- **Nâng cấp toàn diện đáp ứng môi trường thực tế:**
  1. **Động cơ YOLOv8 Real-time Multi-Object Detection:**
     - Khoanh vùng Bounding Box và đếm đa vật thể rác cùng lúc trên luồng webcam thời gian thực (<10ms).
     - Phân luồng màu sắc trực quan (Xanh lá = Tái chế, Cam = Hữu cơ, Đỏ = Nguy hại/Điện tử, Xám = Vô cơ).
     - Tự động phát chuông/cảnh báo banner đỏ khi phát hiện Pin nguy hại hoặc thiết bị điện tử.
  2. **Bộ dữ liệu 12 Lớp Chuẩn Đô Thị (15.520 ảnh):**
     - Bổ sung **Pin (`battery`)** và **Rác Hữu Cơ (`biological`)**, giải quyết trọn vẹn 100% dòng rác sinh hoạt.
     - Mô hình đạt chuẩn sẵn sàng lắp đặt vào thùng rác thông minh ngoài đời thực.
- **Visual:** Ảnh chụp giao diện Web Tab 1 (YOLOv8 Bounding Boxes) và Tab 2 (12-Class Softmax).
- **Lời thoại thuyết trình:**
  > *"Đặc biệt, không dừng lại ở mức mô hình thí nghiệm, nhóm đã nâng cấp hệ thống lên phiên bản WasteCV Station v2.0 Production-grade. Chúng em tích hợp thêm động cơ YOLOv8 giúp khoanh vùng và đếm nhiều loại rác cùng lúc trên camera, đồng thời mở rộng bộ dữ liệu lên 12 lớp với hơn 15.000 ảnh để nhận diện chính xác Pin nguy hại và Rác hữu cơ sinh hoạt. Đây là giải pháp hoàn chỉnh sẵn sàng đưa vào thương mại hóa thực tế."*

---

### SLIDE 15: KẾT LUẬN & TỔNG KẾT ĐỒ ÁN
- **Kết luận:**
  - Hoàn thành xuất sắc mục tiêu đề tài: Vận dụng trọn vẹn lý thuyết Chương 6 Học máy NEU vào bài toán thực tiễn.
  - Xây dựng thành công hệ sinh thái kép: Deep CNN phân loại chuyên sâu 12 lớp + YOLOv8 quét đa vật thể thời gian thực.
  - Sản phẩm phần mềm chạy thật 100%, mã nguồn đã công khai trên GitHub, phục vụ mục tiêu phát triển bền vững và kinh tế tuần hoàn.
- **Lời cảm ơn & Khởi động phần Q&A:**
  - *"Nhóm xin chân thành cảm ơn Thầy Cô đã lắng nghe và rất mong nhận được những góp ý quý báu!"*
- **Lời thoại thuyết trình:**
  > *"Tóm lại, đồ án WasteCV đã chứng minh sức mạnh của mạng nơ-ron sâu trong việc tự động hóa một tác vụ xã hội ý nghĩa. Hệ thống của chúng em đã được kiểm thử, tối ưu trên GPU và public mã nguồn lên GitHub. Chúng em xin chân thành cảm ơn Thầy Cô đã lắng nghe và sẵn sàng trả lời các câu hỏi phản biện ạ!"*
