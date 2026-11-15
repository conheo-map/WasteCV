"""
app.py — Hệ thống Nhận diện và Phân loại Rác thải Công nghiệp (WasteCV Station v3.0 Industrial Edition)
Tích hợp kiến trúc 2 tầng công nghệ tiên tiến nhất:
  1. 🌐 YOLO-World Open-Vocabulary Engine (32+ loại rác phức tạp đời thực không giới hạn lớp).
  2. 🏭 Industrial Contamination & Material Auditing Engine (Giám định mã nhựa RIC #1-#7, mức độ nhiễm bẩn, Luật BVMT 2020).
  3. ⚡ YOLOv8 12 Lớp Rác Chuyên Dụng (Băng chuyền siêu tốc).
  4. 🔬 SOTA Deep CNN 12-Class & 6-Class (Phân loại đơn vật thể chuyên sâu).
"""

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import time
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

# Cấu hình trang giao diện
st.set_page_config(
    page_title="WasteCV Station v3.0 — Phân Loại Rác Công Nghiệp",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ── CƠ SỞ TRI THỨC 12 LỚP RÁC THỰC TẾ ──
CLASSES_12_INFO = {
    "battery": {
        "name_vi": "Pin & Rác Nguy Hại",
        "bin": "Thùng Rác Nguy Hại (Màu Đỏ)",
        "icon": "🔋",
        "color": "#E74C3C",
        "guide": "⚠️ Rác nguy hại! Tuyệt đối không vứt chung với rác sinh hoạt. Đưa về điểm thu gom pin chuyên dụng."
    },
    "biological": {
        "name_vi": "Rác Hữu Cơ (Thức ăn thừa)",
        "bin": "Thùng Rác Hữu Cơ (Màu Cam)",
        "icon": "🍎",
        "color": "#E67E22",
        "guide": "Ủ phân vi sinh Compost hoặc làm thức ăn gia súc. Phân hủy sinh học nhanh."
    },
    "brown-glass": {
        "name_vi": "Thủy Tinh Nâu (Chai bia, siro)",
        "bin": "Thùng Tái Chế Thủy Tinh",
        "icon": "🍾",
        "color": "#795548",
        "guide": "Rửa sạch cặn, phân loại riêng theo màu để tái chế thủy tinh chất lượng cao."
    },
    "cardboard": {
        "name_vi": "Bìa Carton",
        "bin": "Thùng Tái Chế Giấy",
        "icon": "📦",
        "color": "#D2B48C",
        "guide": "Gấp phẳng hộp, giữ khô ráo. Thùng ngấm dầu mỡ chuyển rác hữu cơ/đốt."
    },
    "clothes": {
        "name_vi": "Quần Áo Cũ & Vải",
        "bin": "Thùng Thu Gom Vải / Đồ Cũ",
        "icon": "👕",
        "color": "#9B59B6",
        "guide": "Đồ còn lành lặn mang quyên góp từ thiện; vải vụn rách chuyển nhà máy tái chế sợi."
    },
    "green-glass": {
        "name_vi": "Thủy Tinh Xanh Lá (Chai rượu, nước ngọt)",
        "bin": "Thùng Tái Chế Thủy Tinh",
        "icon": "🍾",
        "color": "#27AE60",
        "guide": "Tráng sạch nước thừa, bỏ nút bần/nắp kim loại trước khi ném vào thùng."
    },
    "metal": {
        "name_vi": "Vỏ Lon & Kim Loại (Nhôm, sắt)",
        "bin": "Thùng Tái Chế Kim Loại",
        "icon": "🥫",
        "color": "#95A5A6",
        "guide": "Ép bẹp vỏ lon nhôm để giảm thể tích. Tái chế kim loại tiết kiệm 95% năng lượng."
    },
    "paper": {
        "name_vi": "Giấy Vụn & Sách Báo",
        "bin": "Thùng Tái Chế Giấy",
        "icon": "📄",
        "color": "#3498DB",
        "guide": "Giữ giấy sạch sẽ, không dính dầu mỡ thực phẩm. Gỡ bỏ ghim bấm kim loại nếu có."
    },
    "plastic": {
        "name_vi": "Chai & Hộp Nhựa",
        "bin": "Thùng Tái Chế Nhựa",
        "icon": "🥤",
        "color": "#E67E22",
        "guide": "Tráng sạch nước thừa, tháo nắp và ép dẹp thân chai."
    },
    "shoes": {
        "name_vi": "Giày Dép Cũ",
        "bin": "Thùng Thu Gom Đồ Dùng Cũ",
        "icon": "👟",
        "color": "#795548",
        "guide": "Đôi còn dùng được quyên góp từ thiện; hỏng nặng chuyển tái chế đế cao su."
    },
    "trash": {
        "name_vi": "Rác Vô Cơ Khác (Không tái chế)",
        "bin": "Thùng Rác Chôn Lấp / Vô Cơ",
        "icon": "🗑️",
        "color": "#7F8C8D",
        "guide": "Túi nilon bẩn, đồ gốm sứ vỡ, tã lót đưa đi thiêu đốt/chôn lấp."
    },
    "white-glass": {
        "name_vi": "Thủy Tinh Trong Suốt (Lọ mứt, chai nước)",
        "bin": "Thùng Tái Chế Thủy Tinh",
        "icon": "🥛",
        "color": "#1ABC9C",
        "guide": "Rửa sạch cặn thức ăn. Thủy tinh trong suốt có giá trị tái chế cao nhất."
    }
}

# ── CƠ SỞ TRI THỨC 6 LỚP TRASHNET ──
CLASSES_6_INFO = {
    "cardboard": {"name_vi": "Bìa Carton", "bin": "Thùng Tái Chế Giấy", "icon": "📦", "color": "#D2B48C", "guide": "Gấp phẳng, giữ khô ráo."},
    "glass": {"name_vi": "Thủy Tinh", "bin": "Thùng Tái Chế Thủy Tinh", "icon": "🍾", "color": "#00CED1", "guide": "Tráng sạch, phân loại riêng nắp."},
    "metal": {"name_vi": "Kim Loại", "bin": "Thùng Tái Chế Kim Loại", "icon": "🥫", "color": "#C0C0C0", "guide": "Nén dẹp vỏ lon để tiết kiệm thể tích."},
    "paper": {"name_vi": "Giấy Vụn/Sách Báo", "bin": "Thùng Tái Chế Giấy", "icon": "📄", "color": "#4682B4", "guide": "Giữ khô ráo, không dính dầu mỡ."},
    "plastic": {"name_vi": "Đồ Nhựa", "bin": "Thùng Tái Chế Nhựa", "icon": "🥤", "color": "#FF8C00", "guide": "Tháo nắp, ép bẹp chai trước khi bỏ thùng."},
    "trash": {"name_vi": "Rác Thải Khác", "bin": "Thùng Rác Vô Cơ", "icon": "🗑️", "color": "#808080", "guide": "Chôn lấp hoặc thiêu đốt theo quy trình."}
}


@st.cache_resource
def load_all_engines():
    """Nạp tất cả các động cơ AI (YOLO-World + YOLOv8 + Deep CNN Classifiers)."""
    engines = {"classifiers": {}, "detector_world": None, "detector_12": None}

    try:
        from detector import WasteDetector
    except ImportError:
        from src.detector import WasteDetector

    # 1. Nạp YOLO-World Open-Vocabulary Detector
    try:
        engines["detector_world"] = WasteDetector(model_type="world")
    except Exception as e:
        st.sidebar.warning(f"Lỗi nạp YOLO-World: {e}")

    # 2. Nạp YOLOv8 12 Lớp Rác Chuyên Dụng
    try:
        engines["detector_12"] = WasteDetector(model_type="waste_12")
    except Exception as e:
        st.sidebar.warning(f"Lỗi nạp YOLOv8 12-class: {e}")

    # 3. Nạp Mô hình 12 Lớp SOTA
    cnn12_path = MODELS_DIR / "best_waste12_cnn.pt"
    if cnn12_path.exists():
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights = models.MobileNet_V2_Weights.DEFAULT
            m12 = models.mobilenet_v2(weights=weights)
            in_features = m12.classifier[1].in_features
            m12.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 12),
            )
            ckpt = torch.load(cnn12_path, map_location=device)
            m12.load_state_dict(ckpt["model_state_dict"])
            m12.to(device).eval()
            engines["classifiers"]["12-Class SOTA Deep CNN (MobileNetV2 - 96.01%)"] = {
                "model": m12, "device": device, "type": "torch_12", "classes": ckpt.get("classes", list(CLASSES_12_INFO.keys()))
            }
        except Exception as e:
            st.sidebar.warning(f"Lỗi nạp mô hình 12 lớp: {e}")

    # 4. Nạp Mô hình 6 Lớp TrashNet
    cnn6_path = MODELS_DIR / "best_cnn.pt"
    if cnn6_path.exists():
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights = models.MobileNet_V2_Weights.DEFAULT
            m6 = models.mobilenet_v2(weights=weights)
            in_f = m6.classifier[1].in_features
            m6.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 6),
            )
            ckpt6 = torch.load(cnn6_path, map_location=device)
            state = ckpt6["model_state_dict"] if "model_state_dict" in ckpt6 else ckpt6
            m6.load_state_dict(state)
            m6.to(device).eval()
            engines["classifiers"]["6-Class Deep CNN (TrashNet - 92.17%)"] = {
                "model": m6, "device": device, "type": "torch_6", "classes": list(CLASSES_6_INFO.keys())
            }
        except Exception as e:
            st.sidebar.warning(f"Lỗi nạp mô hình 6 lớp: {e}")

    # 5. Nạp MLP Baseline
    mlp_path = MODELS_DIR / "mlp_baseline.pkl"
    if mlp_path.exists():
        try:
            mlp_model = joblib.load(mlp_path)
            engines["classifiers"]["MLP Baseline (2 Lớp Ẩn - 46.74%)"] = {
                "model": mlp_model, "type": "sklearn", "classes": list(CLASSES_6_INFO.keys())
            }
        except Exception as e:
            st.sidebar.warning(f"Lỗi nạp MLP: {e}")

    return engines


def predict_cnn_single(model_info, image: Image.Image):
    """Suy luận phân loại đơn ảnh bằng Deep CNN."""
    import torch
    from torchvision import transforms
    device = model_info["device"]
    model = model_info["model"]
    classes = model_info["classes"]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], float(probs[pred_idx]), probs, classes


def main():
    engines = load_all_engines()
    detector_world = engines.get("detector_world")
    detector_12 = engines.get("detector_12")
    classifiers = engines.get("classifiers", {})

    # ── SIDEBAR CẤU HÌNH & TRẠNG THÁI HỆ THỐNG ──
    st.sidebar.title("🎛️ Trung Tâm Điều Khiển")

    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Only"
    except:
        gpu_name = "CPU Only"

    st.sidebar.success(f"⚡ **Phần cứng AI:** {gpu_name}")
    st.sidebar.caption("PyTorch CUDA 12.4 + Automatic Mixed Precision (AMP)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Chọn Động Cơ Phát Hiện (Detection)")
    engine_choice = st.sidebar.radio(
        "Mô hình quét đa vật thể:",
        [
            "🌐 YOLO-World Open-Vocabulary (32+ Loại Rác Thực Tế)",
            "⚡ YOLOv8 12 Lớp Rác Chuyên Dụng (Băng Chuyền Siêu Tốc)"
        ],
        index=0
    )

    if "YOLO-World" in engine_choice and detector_world is not None:
        active_detector = detector_world
        st.sidebar.info("💡 **YOLO-World:** Nhận diện được khẩu trang, ly trà sữa, hộp xốp, vỉ thuốc, pin, vỏ mì tôm...")
        with st.sidebar.expander("➕ Thêm Prompt Rác Tùy Biến (Zero-Shot)"):
            custom_prompt_input = st.text_input("Gõ loại rác tiếng Anh (cách nhau dấu phẩy):", placeholder="lighter, syringe, plastic straw")
            if st.button("Áp dụng Prompts Mới"):
                if custom_prompt_input.strip():
                    new_prompts = [p.strip() for p in custom_prompt_input.split(",") if p.strip()]
                    active_detector.set_prompts(new_prompts)
                    st.success(f"Đã cập nhật {len(new_prompts)} prompts mới!")
    else:
        active_detector = detector_12 if detector_12 is not None else detector_world

    conf_thresh = st.sidebar.slider("Ngưỡng tin cậy (Confidence):", min_value=0.10, max_value=0.90, value=0.20, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Khung Học Thuật & Pháp Lý")
    st.sidebar.info(
        r"""
        - **Pháp lý:** Luật Bảo vệ Môi trường Việt Nam 2020 (Phân loại rác tại nguồn)
        - **Chuẩn nhựa:** ASTM D7611 SPI Resin Identification Code (RIC #1-#7)
        - **Công nghệ AI:** Open-Vocabulary Vision-Language + Transfer Learning SOTA CNN
        """
    )
    st.sidebar.caption("WasteCV Station v3.0 Industrial Edition — 2026")

    # ── TIÊU ĐỀ CHÍNH ──
    st.title("♻️ WasteCV Station: Hệ Thống Phân Loại Rác Công Nghiệp")
    st.markdown(
        "Giải pháp Thị giác Máy tính Công nghiệp phục vụ **Kinh Tế Tuần Hoàn & Đô Thị Thông Minh**. "
        "Tích hợp công nghệ **Open-Vocabulary YOLO-World (32+ loại rác đời thực)** và **Bộ Thẩm Định Mức Độ Nhiễm Bẩn & Mã Nhựa RIC #1-#7**."
    )

    tab1, tab2, tab3 = st.tabs([
        "🏭 Giám Định Công Nghiệp & Quét Thời Gian Thực (Industrial Real-Time AI)",
        "🔬 Phân Loại Chuyên Sâu 12 Lớp (Deep CNN SOTA Classifier)",
        "📊 Bảng Kiểm Toán & Đối Sánh Mô Hình (Audit Analytics)"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: GIÁM ĐỊNH CÔNG NGHIỆP & QUÉT ĐA RÁC (YOLO-WORLD + AUDITOR)
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("🎯 Quét Đa Rác Đời Thực & Thẩm Định Mức Độ Nhiễm Bẩn")
        st.markdown(
            "Phân tích mọi loại rác thải phức tạp ngoài đời thực (kể cả biến dạng, dính bẩn hay dính mỡ), "
            "tự động trích xuất **Mã nhựa quốc tế (RIC #1 đến #7)**, **Đo chỉ số nhiễm bẩn (%)** và chỉ định luồng xử lý theo Luật Môi trường 2020."
        )

        col_in, col_out = st.columns([1, 1], gap="large")

        img_for_yolo = None
        with col_in:
            yolo_mode = st.radio(
                "Nguồn Thu Nhận Khung Hình:",
                ["📸 Chụp từ Webcam (Ảnh tĩnh)", "🎥 Live Stream Tự Động (WebRTC 30 FPS)", "📁 Tải file ảnh hiện trường"],
                horizontal=False,
                key="yolo_input_mode"
            )

            if yolo_mode == "🎥 Live Stream Tự Động (WebRTC 30 FPS)":
                st.info("👉 Bấm **START** bên dưới: Camera sẽ quay liên tục và **TỰ ĐỘNG GIÁM ĐỊNH MỖI KHUNG HÌNH** khi bạn giơ đồ vật lên!")
                try:
                    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
                    import av
                    import queue
                    import threading

                    class AsyncIndustrialVideoProcessor:
                        def __init__(self):
                            self.detector = active_detector
                            self.conf = conf_thresh
                            self.q = queue.Queue(maxsize=1)
                            self.lock = threading.Lock()
                            self.latest_boxes = []
                            self.running = True
                            self.worker = threading.Thread(target=self._worker, daemon=True)
                            self.worker.start()

                        def _worker(self):
                            while self.running:
                                try:
                                    frame_data = self.q.get(timeout=0.1)
                                except queue.Empty:
                                    continue
                                if self.detector is not None:
                                    boxes, _ = self.detector.get_boxes(frame_data, conf_threshold=self.conf, imgsz=288)
                                    with self.lock:
                                        self.latest_boxes = boxes

                        def recv(self, frame):
                            img_bgr = frame.to_ndarray(format="bgr24")
                            if self.q.empty():
                                try:
                                    self.q.put_nowait(img_bgr)
                                except queue.Full:
                                    pass

                            with self.lock:
                                current_boxes = list(self.latest_boxes)

                            if self.detector is not None and current_boxes:
                                annotated_bgr = self.detector.draw_boxes(img_bgr, current_boxes)
                            else:
                                annotated_bgr = img_bgr

                            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

                    webrtc_streamer(
                        key="waste-live-stream-continuous-v4",
                        video_processor_factory=AsyncIndustrialVideoProcessor,
                        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                        media_stream_constraints={
                            "video": {
                                "width": {"ideal": 480, "max": 640},
                                "height": {"ideal": 360, "max": 480},
                                "frameRate": {"ideal": 30, "max": 30}
                            },
                            "audio": False
                        }
                    )
                except Exception as ex:
                    st.warning(f"Lỗi khởi động Live Stream WebRTC: {ex}")

                st.markdown("---")
                col_btn, _ = st.columns([1, 1])
                with col_btn:
                    if st.button("🚀 BẬT CỬA SỔ CAMERA NATIVE 60 FPS (Không Giật Lag)", type="primary"):
                        import subprocess
                        subprocess.Popen([sys.executable, str(BASE_DIR / "src" / "live_camera.py")])
                        st.success("✅ Đã mở cửa sổ Camera Native 60 FPS trực tiếp trên màn hình! Nhấn phím 'q' trên cửa sổ camera để tắt.")

            elif yolo_mode == "📸 Chụp từ Webcam (Ảnh tĩnh)":
                cam_pic = st.camera_input("Quét rác trước camera:")
                if cam_pic:
                    img_for_yolo = Image.open(cam_pic)
            else:
                up_file = st.file_uploader("Chọn ảnh rác hiện trường (JPG, PNG):", type=["jpg", "jpeg", "png"], key="yolo_file_up")
                if up_file:
                    img_for_yolo = Image.open(up_file)

        with col_out:
            if yolo_mode == "🎥 Live Stream Tự Động (WebRTC 30 FPS)":
                st.markdown("#### 🎯 Hướng Dẫn Sử Dụng Live Stream:")
                st.markdown(
                    """
                    1. Bấm **START** ở khung camera bên trái và cấp quyền truy cập Camera.
                    2. **Giơ bất kỳ vật thể nào trước ống kính:**
                       - 🍾 **Chai nước / Vỏ lon / Cốc:** Hệ thống đóng khung và thẩm định mã nhựa.
                       - 😷 **Khẩu trang / Vỉ thuốc:** Hệ thống phát hiện rác y tế và yêu cầu thiêu đốt nhiệt cao.
                       - 🍱 **Hộp xốp bẩn / Đồ ăn dơ:** Hệ thống đo chỉ số dầu mỡ và chuyển sang **Lò đốt RDF**.
                       - 🔋 **Pin / Đồ điện tử:** Hệ thống kích hoạt **BÁO ĐỘNG ĐỎ CÁCH LY**.
                    """
                )
            elif img_for_yolo is None:
                st.info("👈 Hãy chụp ảnh Webcam hoặc tải ảnh bên trái để hệ thống AI giám định chi tiết.")
            elif active_detector is None:
                st.error("Chưa nạp được mô hình nhận diện! Vui lòng kiểm tra thư mục `models/`.")
            else:
                with st.spinner("Đang giám định vật liệu & mức độ ô nhiễm qua Động cơ AI..."):
                    t_start = time.perf_counter()
                    res = active_detector.detect(img_for_yolo, conf_threshold=conf_thresh)
                    det_time = (time.perf_counter() - t_start) * 1000

                # Cảnh báo rác nguy hại nếu phát hiện pin/điện tử
                if res["has_hazard"]:
                    st.error("🚨 **CẢNH BÁO NGUY HẠI CỰC ĐỘ:** Phát hiện Pin / Rác điện tử / Bình áp suất trong khung hình! Yêu cầu cô lập ngay lập tức!")

                # Hiển thị ảnh đã vẽ Bounding Box
                st.image(res["annotated_image"], caption=f"Kết quả quét & thẩm định (Thời gian xử lý: {det_time:.1f} ms)", width=None)

                # Thẻ thống kê KPI Công Nghiệp
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Tổng Vật Thể", res["total_items"])
                kpi2.metric("🟢 Đủ Chuẩn Tái Chế", res["summary"]["Tái Chế"])
                kpi3.metric("⚪ Lò Đốt RDF / Khác", res["summary"]["Đốt RDF / Khác"])
                kpi4.metric("🔴 Nguy Hại Cách Ly", res["summary"]["Nguy Hại"])

                # ── BẢNG HỒ SƠ GIÁM ĐỊNH CÔNG NGHIỆP CHI TIẾT ──
                if res["detections"]:
                    st.markdown("---")
                    st.markdown("### 📋 Hồ Sơ Thẩm Định Vật Liệu & Mức Độ Nhiễm Bẩn (Audit Sheet):")

                    for idx, d in enumerate(res["detections"], 1):
                        aud = d["audit"]
                        with st.expander(f"🔹 Vật thể #{idx}: **{aud['vn_name']}** — Quyết định: `{aud['decision']}`", expanded=True):
                            c_a, c_b = st.columns([1, 1])
                            with c_a:
                                st.markdown(f"- **Tên nhận diện (AI):** `{d['name']}` (Độ tin cậy: **{d['confidence']*100:.1f}%**)")
                                st.markdown(f"- **Mã Nhựa / Vật Liệu:** `{aud['resin_code']}`")
                                st.markdown(f"- **Thành phần cấu tạo:** {aud['material']}")
                                st.markdown(f"- **Luồng chỉ định:** `{aud['target_stream']}`")
                            with c_b:
                                st.markdown(f"**Chỉ số nhiễm bẩn & Dầu mỡ:** `{aud['contamination_score']}%`")
                                st.progress(aud['contamination_score'] / 100.0)
                                if aud['is_hazard']:
                                    st.error(f"⚠️ **Rủi ro:** {aud['hazard_risk']}")
                                else:
                                    st.info(f"ℹ️ **Rủi ro:** {aud['hazard_risk']}")
                                st.markdown(f"👉 **Hướng dẫn theo Luật BVMT 2020:** {aud['action']}")
                else:
                    st.warning("Chưa phát hiện vật thể rõ ràng ở ngưỡng này. Hãy thử giảm thanh trượt Confidence hoặc đưa vật thể lại gần camera hơn.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: PHÂN LOẠI CHUYÊN SÂU (DEEP CNN)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🔬 Phân Loại Đơn Vật Thể Chuyên Sâu (Fine-Grained Classification)")
        st.markdown(
            "Phân tích chuyên sâu hình ảnh đơn vật thể qua mạng nơ-ron tích chập sâu (Deep CNN), "
            "tính toán phân bố xác suất Softmax và đưa ra hướng dẫn xử lý chuẩn ISO."
        )

        col_cnn_in, col_cnn_out = st.columns([1, 1], gap="large")

        selected_clf_name = st.selectbox(
            "Chọn Mô Hình Phân Loại:",
            list(classifiers.keys()) if classifiers else ["Chưa có mô hình"]
        )

        img_for_cnn = None
        with col_cnn_in:
            cnn_mode = st.radio("Nguồn ảnh phân loại:", ["Tải ảnh từ máy tính", "Chụp nhanh từ Webcam"], key="cnn_input_mode")
            if cnn_mode == "Tải ảnh từ máy tính":
                cnn_file = st.file_uploader("Chọn ảnh đơn vật thể (Khuyên dùng vật thể rõ ràng):", type=["jpg", "jpeg", "png"], key="cnn_file_up")
                if cnn_file:
                    img_for_cnn = Image.open(cnn_file)
                    st.image(img_for_cnn, caption="Ảnh vật thể", width=None)
            else:
                cnn_cam = st.camera_input("Chụp ảnh vật thể qua camera:", key="cnn_cam_input")
                if cnn_cam:
                    img_for_cnn = Image.open(cnn_cam)
                    st.image(img_for_cnn, caption="Ảnh chụp webcam", width=None)

        with col_cnn_out:
            if img_for_cnn is None:
                st.info("👈 Hãy tải ảnh hoặc chụp webcam bên trái để mô hình Deep CNN phân tích chuyên sâu.")
            elif selected_clf_name not in classifiers:
                st.error("Mô hình được chọn chưa sẵn sàng.")
            else:
                clf_info = classifiers[selected_clf_name]
                with st.spinner("Đang tính toán phân bố xác suất qua mạng CNN..."):
                    t_start = time.perf_counter()
                    pred_class, pred_conf, probs, class_names = predict_cnn_single(clf_info, img_for_cnn)
                    infer_ms = (time.perf_counter() - t_start) * 1000

                is_12class = (clf_info.get("type") == "torch_12")
                meta = CLASSES_12_INFO.get(pred_class, {}) if is_12class else CLASSES_6_INFO.get(pred_class, {})
                vn_name = meta.get("name_vi", pred_class.capitalize())
                bin_name = meta.get("bin", "Thùng Rác Vô Cơ")
                icon = meta.get("icon", "♻️")
                guide = meta.get("guide", "Tuân thủ hướng dẫn thu gom rác tại địa phương.")

                st.markdown(f"### {icon} Kết Quả: **{vn_name}**")
                st.markdown(f"- **Độ Tin Cậy (Confidence):** `{pred_conf * 100:.2f}%`")
                st.markdown(f"- **Thùng Rác Chỉ Định:** `{bin_name}`")
                st.markdown(f"- **Thời Gian Suy Luận:** `{infer_ms:.2f} ms`")

                if pred_class == "battery":
                    st.error("🚨 **CẢNH BÁO NGUY HẠI:** Đây là Pin hoặc Rác điện tử chứa kim loại nặng độc hại (Chì, Thủy ngân, Cadmium). Tuyệt đối không vứt vào thùng rác thông thường!")
                elif pred_class == "biological":
                    st.success("🌱 **RÁC HỮU CƠ:** Có thể ủ làm phân bón sinh học hoặc biogas!")

                st.info(f"💡 **Hướng Dẫn Xử Lý:** {guide}")

                st.markdown("##### 📊 Phân Bố Xác Suất Chi Tiết Giữa Các Lớp (Top 5):")
                prob_data = pd.DataFrame({
                    "Danh mục": [CLASSES_12_INFO.get(c, {}).get("name_vi", c) if is_12class else CLASSES_6_INFO.get(c, {}).get("name_vi", c) for c in class_names],
                    "Xác suất (%)": [float(p * 100) for p in probs]
                }).sort_values(by="Xác suất (%)", ascending=False).reset_index(drop=True)

                st.dataframe(prob_data.head(5), width=None)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: BẢNG KIỂM TOÁN & ĐỐI SÁNH MÔ HÌNH (AUDIT ANALYTICS)
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("📊 Bảng Kiểm Toán Hiệu Năng & Đối Sánh Đa Mô Hình")
        st.markdown(
            "Đối sánh thực nghiệm khoa học giữa 3 thế hệ mô hình: "
            "**MLP Baseline (Machine Learning)** vs **6-Class Deep CNN (TrashNet)** vs **12-Class SOTA Deep CNN**."
        )

        cm12_path = REPORTS_DIR / "waste12_confusion_matrix.png"
        learn12_path = REPORTS_DIR / "waste12_learning_curves.png"
        cm6_path = REPORTS_DIR / "confusion_matrix.png"
        comp_path = REPORTS_DIR / "model_comparison.png"

        col_rep1, col_rep2 = st.columns([1, 1])
        with col_rep1:
            if cm12_path.exists():
                st.markdown("#### Ma Trận Nhầm Lẫn 12 Lớp (Confusion Matrix SOTA)")
                st.image(str(cm12_path), width=None)
            elif cm6_path.exists():
                st.markdown("#### Ma Trận Nhầm Lẫn 6 Lớp (Confusion Matrix)")
                st.image(str(cm6_path), width=None)

        with col_rep2:
            if learn12_path.exists():
                st.markdown("#### Đường Cong Học Tập 12 Lớp (Learning Curves)")
                st.image(str(learn12_path), width=None)
            elif comp_path.exists():
                st.markdown("#### Biểu Đồ Đối Sánh Hiệu Năng")
                st.image(str(comp_path), width=None)

        st.markdown("---")
        st.markdown("#### 🏆 Bảng So Sánh Các Chỉ Số Học Máy Toàn Diện (Benchmark Summary)")

        comp_df = pd.DataFrame([
            {
                "Mô Hình": "MLP Baseline (2 Hidden Layers)",
                "Dữ Liệu Huấn Luyện": "TrashNet 6 Lớp (Flat Pixels)",
                "Accuracy (Test)": "46.74%",
                "Macro F1-Score": "44.12%",
                "Độ Trễ Suy Luận": "1.2 ms (CPU)",
                "Ưu / Nhược Điểm": "Kém chính xác, mất thông tin không gian ảnh 2D"
            },
            {
                "Mô Hình": "6-Class Deep CNN (MobileNetV2)",
                "Dữ Liệu Huấn Luyện": "TrashNet 6 Lớp (2.527 ảnh)",
                "Accuracy (Test)": "92.17%",
                "Macro F1-Score": "90.85%",
                "Độ Trễ Suy Luận": "3.8 ms (GPU RTX 2050)",
                "Ưu / Nhược Điểm": "Chính xác cao nhưng thiếu rác Pin và Hữu cơ"
            },
            {
                "Mô Hình": "12-Class SOTA Deep CNN (MobileNetV2)",
                "Dữ Liệu Huấn Luyện": "Waste 12 Lớp Thực Tế (15.520 ảnh)",
                "Accuracy (Test)": "96.01%",
                "Macro F1-Score": "94.76%",
                "Độ Trễ Suy Luận": "4.2 ms (GPU RTX 2050)",
                "Ưu / Nhược Điểm": "SOTA! Recall Pin đạt 100%, có rác Hữu cơ sinh học"
            },
            {
                "Mô Hình": "YOLO-World Open-Vocabulary + Auditor",
                "Dữ Liệu Huấn Luyện": "Zero-Shot Text Prompting (32+ lớp)",
                "Accuracy (Test)": "Thời gian thực đa rác",
                "Macro F1-Score": "Đo độ bẩn & mã nhựa",
                "Độ Trễ Suy Luận": "18.5 ms (GPU RTX 2050)",
                "Ưu / Nhược Điểm": "Đỉnh cao! Nhận diện rác đời thực không giới hạn"
            }
        ])

        st.dataframe(comp_df, width=None)


if __name__ == "__main__":
    main()
