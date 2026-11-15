# 🎯 BỘ CÂU HỎI & ĐÁP ÁN PHẢN BIỆN BẢO VỆ ĐỒ ÁN AI
## ĐỀ TÀI: WASTECV — HỆ THỐNG PHÂN LOẠI RÁC THẢI BẰNG THỊ GIÁC MÁY TÍNH
> Bám sát bài giảng **Chương 6: Học Máy (Machine Learning) — ĐH Kinh Tế Quốc Dân (NEU)**  
> Dành cho sinh viên tự tin đạt điểm A/A+ khi bảo vệ trước Hội đồng chấm thi.

---

### CÂU HỎI 1: Em hãy giải thích bộ ba (T, P, E) của Tom Mitchell áp dụng cụ thể vào đề tài này như thế nào?
- **Vị trí bài giảng:** Slide 11 Chương 6 (Định nghĩa Machine Learning của Tom Mitchell 1997).
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, theo định nghĩa của Tom Mitchell tại Slide 11 bài giảng:*
  > - *Nhiệm vụ **T (Task)** của hệ thống là: Tiếp nhận một ảnh đầu vào chứa vật thể rác thải và dự đoán nhãn phân loại tương ứng trong 6 lớp: Cardboard, Glass, Metal, Paper, Plastic, hoặc Trash.*
  > - *Thước đo hiệu năng **P (Performance)** là: Độ chính xác tổng thể (Accuracy), độ chuẩn xác (Precision), độ bao phủ (Recall), điểm điều hòa F1-Score tính trên tập kiểm tra độc lập (Test Set), cùng với độ trễ suy luận thời gian thực (Latency < 30ms).*
  > - *Kinh nghiệm **E (Experience)** là: Quá trình mô hình được huấn luyện lặp đi lặp lại trên tập dữ liệu TrashNet gồm 1.766 ảnh gắn nhãn thực tế, qua đó cập nhật các ma trận trọng số liên kết $W$ thông qua giải thuật lan truyền ngược Backpropagation."*

---

### CÂU HỎI 2: Tại sao mô hình MLP Baseline chỉ đạt 46.74% trong khi mô hình MobileNetV2 CNN lại đạt tới 92.17%? Bản chất toán học và kiến trúc khác nhau ở điểm nào?
- **Vị trí bài giảng:** Slide 36–45 (Mô hình nơ-ron sinh học và nhân tạo, mạng nhiều lớp MLP) và kiến thức mở rộng Mạng Tích chập (CNN).
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, sự chênh lệch hơn 45% độ chính xác bắt nguồn từ 2 nguyên nhân cốt lõi:*
  > 1. *Thứ nhất là việc **bảo toàn tương quan không gian (Spatial Topology)**: Mạng MLP yêu cầu duỗi phẳng ảnh $64 \times 64 \times 3$ thành một vector 1D ($12.288$ chiều). Việc này làm mất hoàn toàn mối liên hệ không gian giữa các pixel cạnh nhau (ví dụ: đường biên của chiếc vỏ lon hay độ cong của miệng chai). Ngược lại, các tầng tích chập của CNN sử dụng các bộ lọc ma trận (Kernels) trượt trên ma trận 2D, giữ nguyên vẹn cấu trúc hình học và tự động học được các đặc trưng từ thấp (cạnh, góc) đến cao (kết cấu, hình khối).*
  > 2. *Thứ hai là nguyên lý **Chia sẻ trọng số (Weight Sharing)** và **Bất biến dịch chuyển (Translation Invariance)**: Trong MLP, một nơ-ron chỉ kết nối với một vị trí cố định của pixel. Nếu vỏ lon nằm lệch sang góc trái hay góc phải, MLP sẽ nhận diện sai. Trong khi đó, các bộ lọc của CNN quét qua toàn bộ ảnh nên nhận diện được vật thể dù nó nằm ở bất kỳ vị trí nào trong khung hình."*

---

### CÂU HỎI 3: Tại sao trong bảng kết quả của MLP, lớp Kim loại (Metal) lại có Precision = 100% nhưng Recall chỉ đạt 1.61%? Hiện tượng này nói lên điều gì?
- **Vị trí bài giảng:** Slide 25 (Độ đo đánh giá phân lớp, Ma trận nhầm lẫn, Trade-off giữa Precision và Recall).
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, đây là một điểm thực nghiệm rất đắt giá thể hiện sự đánh đổi (trade-off) giữa Precision và Recall:*
  > - *Recall = $\frac{TP}{TP + FN} = \frac{1}{1 + 61} = 1.61\%$: Trong tập kiểm tra có 62 vỏ kim loại, nhưng MLP chỉ tìm ra duy nhất 1 vỏ kim loại, bỏ sót tới 61 vỏ lon (FN = 61).*
  > - *Precision = $\frac{TP}{TP + FP} = \frac{1}{1 + 0} = 100\%$: Duy nhất 1 lần mô hình dự đoán là kim loại thì lần đó lại đoán đúng, nên độ chuẩn xác đạt 100%.*
  > *Hiện tượng này chứng minh mô hình MLP bị **Kém khớp (Underfitting)** nghiêm trọng trên lớp Metal; mô hình quá dè dặt và thiếu tự tin, hầu như gán các vỏ lon kim loại sang lớp Plastic hoặc Paper. Nếu chúng ta chỉ nhìn vào con số Precision 100% mà không xem xét Recall và F1-Score ($3.17\%$), chúng ta sẽ bị ngộ nhận về hiệu năng của mô hình."*

---

### CÂU HỎI 4: Quá khớp (Overfitting) là gì? Trong đồ án này, nhóm đã sử dụng những cơ chế nào để ngăn chặn Overfitting?
- **Vị trí bài giảng:** Slide 28 (Hiện tượng dừng sớm - Early Stopping) & Slide 31 (Quá khớp - Overfitting).
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, theo Slide 31, Quá khớp (Overfitting) là hiện tượng mô hình học quá kỹ và 'thuộc lòng' cả những đặc trưng nhiễu hoặc ngẫu nhiên của tập huấn luyện, dẫn đến việc sai số trên tập Train rất thấp nhưng sai số trên tập dữ liệu mới (Validation/Test) lại tăng cao.*
  > *Để kiểm soát Overfitting, nhóm áp dụng 4 kỹ thuật:*
  > 1. *Chia dữ liệu Stratified thành 3 tập riêng biệt: Train (70%), Val (15%), Test (15%) và chỉ chấm điểm mô hình trên tập Test sau khi đã hoàn tất huấn luyện.*
  > 2. *Áp dụng cơ chế **Dừng sớm (Early Stopping - Slide 28)**: Theo dõi Validation Loss sau mỗi epoch; nếu sau 5 epoch liên tiếp loss không giảm thì hệ thống tự động ngắt huấn luyện và khôi phục checkpoint tốt nhất.*
  > 3. *Kỹ thuật **Dropout Regularization**: Ngắt kết nối ngẫu nhiên 30% và 20% các nơ-ron trong Classifier head để triệt tiêu sự phụ thuộc chéo giữa các đặc trưng.*
  > 4. *Kỹ thuật **Tăng cường dữ liệu (Data Augmentation)**: Xoay ảnh ngẫu nhiên, lật ngang, đổi độ sáng để ép mô hình học đặc trưng bản chất của rác thay vì học thuộc phông nền."*

---

### CÂU HỎI 5: Em hãy trình bày nguyên lý toán học của thuật toán Lan truyền ngược (Backpropagation) và quy tắc cập nhật Gradient Descent?
- **Vị trí bài giảng:** Slide 55 (Thuật toán tối ưu Gradient Descent) & Slide 60–62 (Học mạng nhiều lớp - Lan truyền ngược sai số).
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, quá trình học trong mạng nơ-ron được thực hiện qua 2 bước lặp tuần hoàn:*
  > - *Bước 1: **Lan truyền tiến (Forward Pass)**: Dữ liệu ảnh đi qua các tầng nơ-ron, tính $\text{Net} = \sum w_j \cdot x_j + b$ và qua hàm kích hoạt $\text{Out} = g(\text{Net})$. Tại tầng ra cuối cùng, hàm Softmax tính xác suất và hàm Cross-Entropy đo lường sai số $\mathcal{L}$ giữa nhãn thực tế $y$ và dự đoán $\hat{y}$.*
  > - *Bước 2: **Lan truyền ngược (Backward Pass - Slide 60)**: Sử dụng quy tắc chuỗi (Chain Rule) của giải tích vi phân để tính đạo hàm riêng của hàm mất mát đối với từng trọng số: $\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial \text{Out}} \cdot \frac{\partial \text{Out}}{\partial \text{Net}} \cdot \frac{\partial \text{Net}}{\partial w_{ij}}$. Sai số được truyền ngược từ lớp ra về các lớp ẩn.*
  > - *Sau đó, trọng số được cập nhật theo hướng ngược chiều gradient: $w_{ij} \leftarrow w_{ij} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}}$, trong đó $\eta$ là tốc độ học (Learning Rate) như nêu tại Slide 55."*

---

### CÂU HỎI 6: Tại sao ở lớp đầu ra nhóm dùng hàm Softmax mà không dùng Sigmoid hay Linear?
- **Vị trí bài giảng:** Slide 44 (Hàm Sigmoid) & Lý thuyết bài toán phân loại đa lớp.
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô:*
  > - *Hàm **Linear** không bị chặn, đầu ra có thể là một số âm hoặc số vô cùng lớn, không mang ý nghĩa xác suất.*
  > - *Hàm **Sigmoid** $g(z) = \frac{1}{1 + e^{-z}}$ chỉ phù hợp cho bài toán phân loại nhị phân (2 lớp) hoặc bài toán gán nhiều nhãn độc lập (Multi-label), vì tổng xác suất qua Sigmoid của 6 lớp không bắt buộc phải bằng 1.*
  > - *Trong khi đó, bài toán của chúng em là **Phân loại đa lớp loại trừ lẫn nhau (Multiclass Single-label)** — một vật thể tại một thời điểm chỉ có thể thuộc về duy nhất 1 loại rác. Hàm **Softmax**:*
  >   $$P(y = c \mid x) = \frac{e^{z_c}}{\sum_{k=1}^{6} e^{z_k}}$$
  >   *vừa đảm bảo đưa mọi giá trị về khoảng $(0, 1)$, vừa đảm bảo tổng xác suất của cả 6 lớp luôn bằng đúng $1.0$ (tương đương $100\%$), giúp bộ điều phối dễ dàng chọn ra nhãn có xác suất cao nhất."*

---

### CÂU HỎI 7: Dựa vào Ma trận nhầm lẫn (Confusion Matrix), các lỗi phân loại chính của mô hình nằm ở đâu? Tại sao lại xảy ra hiện tượng đó?
- **Vị trí bài giảng:** Slide 25 (Ma trận nhầm lẫn - Confusion Matrix).
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, phân tích ma trận nhầm lẫn của mô hình MobileNetV2 cho thấy:*
  > - *Các lớp Cardboard và Paper có độ chính xác gần như tuyệt đối (Precision 100% và 97.7%) do kết cấu sợi giấy và màu nâu carton rất đặc trưng.*
  > - *Lỗi nhầm lẫn đáng kể nhất xuất hiện giữa **Nhựa (Plastic)** và **Thủy tinh (Glass)**: Có 11 mẫu chai nhựa bị mô hình dự đoán nhầm thành chai thủy tinh.*
  > - *Nguyên nhân thị giác là: Cả hai loại rác này đều là chai lọ hình trụ thuôn dài, cùng có đặc tính trong suốt và phản chiếu vệt sáng trắng trên bề mặt dưới ánh đèn. Để khắc phục triệt để trong tương lai, hệ thống cần tích hợp thêm cảm biến trọng lượng hoặc phân tích quang phổ hồng ngoại gần (NIR)."*

---

### CÂU HỎI 8: Tập dữ liệu TrashNet có lớp 'Trash' chỉ có 137 ảnh (chiếm 5.4%), tức là bị mất cân bằng dữ liệu (Class Imbalance). Nhóm đã xử lý vấn đề này như thế nào?
- **Vị trí bài giảng:** Khái niệm Mất cân bằng dữ liệu & Stratified Sampling.
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, lớp 'Trash' đại diện cho rác hỗn hợp khó tái chế (túi nilon rách, xốp dơ), có số lượng mẫu rất khiêm tốn ($137$ ảnh so với gần $600$ ảnh giấy). Nếu không xử lý, mô hình sẽ có xu hướng bỏ qua lớp này.*
  > *Nhóm đã áp dụng 3 giải pháp đồng bộ:*
  > 1. *Phân chia dữ liệu bằng **Stratified Split**: Đảm bảo tỷ lệ 5.4% được giữ nguyên đồng nhất trong cả tập Train, Val và Test.*
  > 2. *Tăng cường dữ liệu (Data Augmentation) mạnh hơn cho các ảnh rác hỗn hợp trong quá trình sinh batch của DataLoader.*
  > 3. *Đánh giá bằng **Macro-averaged F1-Score**: Tính trung bình không trọng số giữa các lớp để lớp 'Trash' có tiếng nói ngang bằng với các lớp nhiều mẫu. Kết quả thực nghiệm cho thấy mô hình MobileNetV2 đạt **Recall lên tới 95.24%** trên lớp Trash (bắt trúng 20/21 mẫu kiểm tra), hoàn toàn không bị hiện tượng bỏ sót."*

---

### CÂU HỎI 9: Hệ thống này có thể đưa vào thực tế như thế nào? Phần cứng tối thiểu để triển khai là gì?
- **Vị trí bài giảng:** Ứng dụng thực tiễn của Trí Tuệ Nhân Tạo.
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, mô hình MobileNetV2 được thiết kế chuyên biệt cho thiết bị di động và phần cứng biên (Edge Devices):*
  > - *Dung lượng file mô hình chỉ vỏn vẹn **9.8 MB**, mức tiêu thụ bộ nhớ RAM khi chạy dưới 150 MB.*
  > - *Độ trễ suy luận trên GPU laptop chỉ **6.4 ms**; khi nạp vào vi máy tính giá rẻ như **Raspberry Pi 4 hoặc Jetson Orin Nano** sử dụng ONNX Runtime, tốc độ đạt khoảng 25-30 ms/ảnh.*
  > - *Hệ thống có thể gắn trực tiếp trên nóc các thùng rác thông minh tại trường học, bệnh viện hoặc trung tâm thương mại. Khi người dùng đưa rác đến trước mắt camera, vi điều khiển (như Arduino) sẽ điều khiển động cơ servo tự động mở nắp thùng rác tái chế hoặc thùng rác vô cơ tương ứng."*

---

### CÂU HỎI 10: Nếu có thêm thời gian phát triển đồ án, nhóm sẽ cải tiến những điểm gì tiếp theo?
- **Vị trí bài giảng:** Hướng nghiên cứu mở rộng trong Thị giác Máy tính và AI.
- **Cách trả lời chuẩn chỉnh:**
  > *"Thưa Thầy Cô, nếu có thêm thời gian, nhóm sẽ tập trung vào 2 đột phá:*
  > 1. *Chuyển đổi từ bài toán **Phân loại ảnh (Image Classification)** sang bài toán **Phát hiện vật thể (Object Detection)** bằng các kiến trúc như YOLOv8/v11. Điều này sẽ cho phép hệ thống nhận diện và đếm đồng thời nhiều loại rác cùng nằm trong một khay phân loại hoặc trên băng chuyền.*
  > 2. *Mở rộng tập dữ liệu với các loại rác thải sinh hoạt đặc thù của Việt Nam (như hộp xốp đựng cơm, ly trà sữa, túi nilon mỏng) để tăng độ thích ứng khi triển khai vào đời sống thực tế."*
