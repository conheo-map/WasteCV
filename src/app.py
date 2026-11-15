"""
app.py — Ứng dụng Web Demo Phân loại Rác thải Thông minh (Smart Waste Sorting Demo)
Sử dụng Streamlit + OpenCV + PyTorch / Scikit-learn
Hỗ trợ 2 chế độ: Tải ảnh lên (Upload) hoặc Chụp trực tiếp từ Webcam laptop.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path

# Thiết lập tiêu đề trang và layout
st.set_page_config(
    page_title="WasteCV — Hệ Thống Phân Loại Rác Thông Minh",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

CLASSES_INFO = {
    "cardboard": {
        "name_vi": "Bìa Carton",
        "bin": "Thùng Tái Chế (Màu Trắng/Xanh)",
        "icon": "📦",
        "color": "#D2B48C",
        "guide": "Gấp gọn phẳng để tiết kiệm diện tích. Giữ khô ráo, không dính dầu mỡ."
    },
    "glass": {
        "name_vi": "Chai / Lọ Thủy Tinh",
        "bin": "Thùng Tái Chế (Khu vực đồ dễ vỡ)",
        "icon": "🍾",
        "color": "#00CED1",
        "guide": "Tráng sạch cặn bẩn bên trong. Bỏ nắp kim loại/nhựa riêng biệt."
    },
    "metal": {
        "name_vi": "Kim Loại (Vỏ lon, Đồ hộp)",
        "bin": "Thùng Tái Chế (Kim loại)",
        "icon": "🥫",
        "color": "#C0C0C0",
        "guide": "Nén dẹp vỏ lon để tiết kiệm thể tích. Kim loại có giá trị tái chế 100%."
    },
    "paper": {
        "name_vi": "Giấy (Sách báo, Giấy vụn)",
        "bin": "Thùng Tái Chế (Giấy)",
        "icon": "📄",
        "color": "#4682B4",
        "guide": "Giữ giấy khô. Không phân loại giấy ăn bẩn hay hóa đơn nhiệt vào đây."
    },
    "plastic": {
        "name_vi": "Nhựa (Chai PET, Hộp nhựa)",
        "bin": "Thùng Tái Chế (Nhựa)",
        "icon": "🥤",
        "color": "#FF8C00",
        "guide": "Tráng sạch nước thừa, tháo nắp và ép phẳng trước khi vứt."
    },
    "trash": {
        "name_vi": "Rác Thải Khác (Không tái chế)",
        "bin": "Thùng Rác Vô Cơ / Chôn Lấp",
        "icon": "🗑️",
        "color": "#808080",
        "guide": "Rác thải hỗn hợp, túi nilon dơ, xốp bẩn cần đưa đi xử lý thiêu đốt/chôn lấp."
    }
}

CLASS_KEYS = list(CLASSES_INFO.keys())


@st.cache_resource
def load_models():
    """Tải các mô hình đã học vào bộ nhớ."""
    models_dict = {}

    # 1. Thử tải MobileNetV2
    cnn_path = MODELS_DIR / "best_cnn.pt"
    if cnn_path.exists():
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights = models.MobileNet_V2_Weights.DEFAULT
            cnn_model = models.mobilenet_v2(weights=weights)
            in_features = cnn_model.classifier[1].in_features
            cnn_model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, len(CLASS_KEYS)),
            )
            checkpoint = torch.load(cnn_path, map_location=device)
            if "model_state_dict" in checkpoint:
                cnn_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                cnn_model.load_state_dict(checkpoint)
            cnn_model.to(device).eval()
            models_dict["MobileNetV2 Deep CNN"] = {"model": cnn_model, "device": device, "type": "torch"}
        except Exception as e:
            st.sidebar.warning(f"Chưa nạp được MobileNetV2: {e}")

    # 2. Thử tải MLP Baseline
    mlp_path = MODELS_DIR / "mlp_baseline.pkl"
    if mlp_path.exists():
        try:
            mlp_model = joblib.load(mlp_path)
            models_dict["MLP Baseline (2 Hidden Layers)"] = {"model": mlp_model, "type": "sklearn"}
        except Exception as e:
            st.sidebar.warning(f"Chưa nạp được MLP: {e}")

    return models_dict


def predict_cnn(model_info, image: Image.Image):
    """Suy luận bằng mô hình Deep CNN."""
    import torch
    from torchvision import transforms
    device = model_info["device"]
    model = model_info["model"]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    return CLASS_KEYS[pred_idx], float(probabilities[pred_idx]), probabilities


def predict_mlp(model_info, image: Image.Image):
    """Suy luận bằng mô hình MLP Baseline."""
    model = model_info["model"]
    img_resized = image.convert("RGB").resize((64, 64))
    feat = np.array(img_resized, dtype=np.float32).flatten().reshape(1, -1) / 255.0

    probs = model.predict_proba(feat)[0]
    pred_idx = int(np.argmax(probs))
    return CLASS_KEYS[pred_idx], float(probs[pred_idx]), probs


def main():
    # ── SIDEBAR CẤU HÌNH & LÝ THUYẾT ──
    st.sidebar.title("⚙️ Cấu Hình Hệ Thống")
    models = load_models()

    if not models:
        st.sidebar.error("⚠️ Chưa tìm thấy file mô hình trong `models/`. Hãy chạy `train_mlp.py` hoặc `train_cnn.py` trước!")
        selected_model_name = "Chưa có mô hình"
    else:
        selected_model_name = st.sidebar.selectbox("Chọn Mô hình Suy luận:", list(models.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Khung Lý Thuyết (Chương 6)")
    st.sidebar.info(
        r"""
        - **Dạng bài toán:** Học có giám sát — Phân lớp (Slide 14)
        - **Bộ ba (T, P, E):** T=Phân loại, P=Accuracy/F1, E=TrashNet (Slide 11)
        - **Mô hình Nơ-ron:** $\text{Net} = \sum w_j \cdot s_j$, $\text{out} = g(\text{Net})$ (Slide 37)
        - **Học có giám sát:** Lan truyền ngược (Backpropagation) (Slide 60–62)
        - **Đánh giá:** Ma trận nhầm lẫn (Confusion Matrix) (Slide 25)
        """
    )
    st.sidebar.caption("Đồ án môn Trí Tuệ Nhân Tạo — NEU 2026")

    # ── GIAO DIỆN CHÍNH ──
    st.title("♻️ WasteCV: Hệ Thống Nhận Diện & Phân Loại Rác Thải")
    st.markdown(
        "Ứng dụng thị giác máy tính và học máy phục vụ tự động hóa phân loại rác thải tại nguồn, "
        "thúc đẩy tái chế và phát triển kinh tế tuần hoàn."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    image_to_process = None

    with col1:
        st.subheader("📷 Thu Thập Hình Ảnh")
        input_mode = st.radio("Chọn Phương thức Đầu vào:", ["Tải file ảnh lên", "Chụp từ Webcam Laptop"], horizontal=True)

        if input_mode == "Tải file ảnh lên":
            uploaded_file = st.file_uploader("Chọn ảnh rác thải (JPG, PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)
                st.image(image_to_process, caption="Ảnh đã chọn", use_container_width=True)
        else:
            camera_image = st.camera_input("Chụp ảnh rác trước camera:")
            if camera_image is not None:
                image_to_process = Image.open(camera_image)
                st.image(image_to_process, caption="Ảnh chụp từ webcam", use_container_width=True)

    with col2:
        st.subheader("🧠 Kết Quả Phân Loại & Điều Phối")

        if image_to_process is None:
            st.info("👈 Hãy tải ảnh hoặc chụp ảnh từ webcam bên trái để hệ thống nhận diện.")
        elif selected_model_name not in models:
            st.warning("Vui lòng huấn luyện ít nhất 1 mô hình để bắt đầu suy luận.")
        else:
            model_info = models[selected_model_name]

            with st.spinner("Đang xử lý khung hình và suy luận..."):
                t0 = time.perf_counter()
                if model_info["type"] == "torch":
                    pred_class, confidence, all_probs = predict_cnn(model_info, image_to_process)
                else:
                    pred_class, confidence, all_probs = predict_mlp(model_info, image_to_process)
                latency_ms = (time.perf_counter() - t0) * 1000

            info = CLASSES_INFO[pred_class]

            # Khối kết quả nổi bật
            st.markdown(
                f"""
                <div style="background-color: {info['color']}22; border-left: 6px solid {info['color']}; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="margin: 0; color: #111;">{info['icon']} {info['name_vi']} ({pred_class.upper()})</h2>
                    <p style="font-size: 18px; margin: 8px 0 0 0;">
                        <b>Độ tin cậy:</b> <span style="color: {info['color']}; font-weight: bold;">{confidence * 100:.1f}%</span> | 
                        <b>Độ trễ:</b> {latency_ms:.1f} ms
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(confidence))

            st.markdown("#### 🎯 Hướng Dẫn Phân Loại Thùng Rác")
            st.success(f"**Nơi xử lý:** {info['bin']}")
            st.info(f"💡 **Khuyến nghị:** {info['guide']}")

            # Biểu đồ phân bố xác suất Softmax
            st.markdown("#### 📊 Phân Bố Xác Suất 6 Lớp (Softmax Output)")
            chart_data = {CLASSES_INFO[c]["name_vi"]: float(all_probs[i]) * 100 for i, c in enumerate(CLASS_KEYS)}
            st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
