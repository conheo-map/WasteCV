"""
detector.py — Bộ phát hiện và khoanh vùng đa vật thể rác thải thời gian thực (Real-time YOLOv8 Waste Detector)
Sử dụng mô hình YOLOv8n kết hợp bản đồ phân loại rác thải 4 luồng tái chế:
  - 🟢 Xanh lá: Rác Tái Chế (Chai nhựa, Thủy tinh, Vỏ lon, Bìa giấy, Hộp carton)
  - 🟠 Cam:    Rác Hữu Cơ / Đồ ăn thừa (Trái cây, Thực phẩm, Bánh mì, Rau củ)
  - 🔴 Đỏ:     Rác Nguy Hại / Rác Điện Tử (Pin, Điện thoại, Chuột, Bàn phím)
  - ⚪ Xám:    Rác Khác / Vô Cơ (Túi bẩn, Đồ dùng hỗn hợp)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"

# Bản đồ phân loại vật thể nhận diện sang 4 luồng rác thải thực tế
WASTE_CATEGORY_MAP = {
    # 1. RÁC TÁI CHẾ (RECYCLABLE)
    "bottle": {"stream": "Tái Chế (Nhựa / Thủy Tinh)", "color": (46, 204, 113), "bin": "Thùng Tái Chế"},
    "cup": {"stream": "Tái Chế (Ly / Cốc)", "color": (46, 204, 113), "bin": "Thùng Tái Chế"},
    "wine glass": {"stream": "Tái Chế (Thủy Tinh)", "color": (46, 204, 113), "bin": "Thùng Tái Chế"},
    "bowl": {"stream": "Tái Chế (Bát / Hộp nhựa)", "color": (46, 204, 113), "bin": "Thùng Tái Chế"},
    "book": {"stream": "Tái Chế (Sách báo / Giấy)", "color": (52, 152, 219), "bin": "Thùng Tái Chế"},
    "fork": {"stream": "Tái Chế (Dụng cụ kim loại/nhựa)", "color": (46, 204, 113), "bin": "Thùng Tái Chế"},
    "knife": {"stream": "Tái Chế (Dụng cụ kim loại)", "color": (46, 204, 113), "bin": "Thùng Tái Chế"},
    "spoon": {"stream": "Tái Chế (Dụng cụ kim loại)", "color": (46, 204, 113), "bin": "Thùng Tái Chế"},

    # 2. RÁC HỮU CƠ / THỰC PHẨM (ORGANIC / COMPOST)
    "banana": {"stream": "Hữu Cơ (Vỏ chuối)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "apple": {"stream": "Hữu Cơ (Trái táo)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "sandwich": {"stream": "Hữu Cơ (Thức ăn thừa)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "orange": {"stream": "Hữu Cơ (Vỏ cam)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "broccoli": {"stream": "Hữu Cơ (Rau củ)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "carrot": {"stream": "Hữu Cơ (Cà rốt)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "hot dog": {"stream": "Hữu Cơ (Thức ăn thừa)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "pizza": {"stream": "Hữu Cơ (Thức ăn thừa)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "donut": {"stream": "Hữu Cơ (Bánh ngọt)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},
    "cake": {"stream": "Hữu Cơ (Bánh ngọt)", "color": (230, 126, 34), "bin": "Thùng Rác Hữu Cơ"},

    # 3. RÁC NGUY HẠI / ĐIỆN TỬ (HAZARDOUS / E-WASTE)
    "cell phone": {"stream": "Nguy Hại / E-Waste (Điện thoại/Pin)", "color": (231, 76, 60), "bin": "Thùng Rác Nguy Hại"},
    "laptop": {"stream": "E-Waste (Máy tính xách tay)", "color": (231, 76, 60), "bin": "Thùng Rác Điện Tử"},
    "mouse": {"stream": "E-Waste (Chuột máy tính)", "color": (231, 76, 60), "bin": "Thùng Rác Điện Tử"},
    "keyboard": {"stream": "E-Waste (Bàn phím)", "color": (231, 76, 60), "bin": "Thùng Rác Điện Tử"},
    "remote": {"stream": "E-Waste / Pin (Điều khiển)", "color": (231, 76, 60), "bin": "Thùng Rác Nguy Hại"},
    "tv": {"stream": "E-Waste (Màn hình điện tử)", "color": (231, 76, 60), "bin": "Thùng Rác Điện Tử"},

    # 4. RÁC VẢI / PHỤ KIỆN (TEXTILE / OTHERS)
    "backpack": {"stream": "Tái Chế / Vải (Balo cũ)", "color": (155, 89, 182), "bin": "Thùng Đồ Cũ / Vải"},
    "handbag": {"stream": "Tái Chế / Vải (Túi xách)", "color": (155, 89, 182), "bin": "Thùng Đồ Cũ / Vải"},
    "suitcase": {"stream": "Rác Cồng Kềnh (Vali)", "color": (155, 89, 182), "bin": "Điểm Thu Gom Rác Lớn"},
}


class WasteDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path))
        print(f"[✓] Đã khởi tạo WasteDetector trên thiết bị: {self.device}")

    def detect_bgr(self, bgr_frame, conf_threshold=0.25):
        """
        Nhận diện trực tiếp trên khung hình BGR (cho Live Stream WebRTC/OpenCV).
        Trả về annotated_bgr, detections, summary, has_hazard.
        """
        results = self.model.predict(
            source=bgr_frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        annotated_bgr = bgr_frame.copy()
        detections = []
        summary = {"Tái Chế": 0, "Hữu Cơ": 0, "Nguy Hại": 0, "Khác": 0}
        has_hazard = False

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = results[0].names[cls_id]
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                info = WASTE_CATEGORY_MAP.get(
                    cls_name,
                    {"stream": f"Vật thể ({cls_name})", "color": (149, 165, 166), "bin": "Thùng Rác Vô Cơ"}
                )
                stream = info["stream"]
                color_rgb = info["color"]
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                if "Nguy Hại" in stream or "E-Waste" in stream:
                    summary["Nguy Hại"] += 1
                    has_hazard = True
                elif "Hữu Cơ" in stream:
                    summary["Hữu Cơ"] += 1
                elif "Tái Chế" in stream:
                    summary["Tái Chế"] += 1
                else:
                    summary["Khác"] += 1

                detections.append({
                    "name": cls_name,
                    "stream": stream,
                    "bin": info["bin"],
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                })

                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color_bgr, 3)
                label_text = f"{stream}: {conf*100:.0f}%"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_bgr, (x1, max(0, y1 - 25)), (x1 + w + 10, y1), color_bgr, -1)
                cv2.putText(annotated_bgr, label_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return annotated_bgr, detections, summary, has_hazard

    def detect(self, image, conf_threshold=0.25):
        """
        Nhận diện đa vật thể rác thải và vẽ bounding boxes phân loại màu sắc.
        Input: PIL.Image hoặc numpy.ndarray (RGB)
        Output:
          - annotated_image (PIL.Image)
          - detections (list các vật thể kèm tọa độ, nhãn, độ tin cậy)
          - summary (thống kê số lượng rác theo từng luồng)
        """
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        else:
            img_rgb = image.copy()

        # Suy luận YOLOv8
        results = self.model.predict(
            source=img_rgb,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        annotated_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        detections = []
        summary = {
            "Tái Chế": 0,
            "Hữu Cơ": 0,
            "Nguy Hại": 0,
            "Khác": 0
        }
        has_hazard = False

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = results[0].names[cls_id]
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Ánh xạ phân loại luồng rác
                info = WASTE_CATEGORY_MAP.get(
                    cls_name,
                    {"stream": f"Vật thể ({cls_name})", "color": (149, 165, 166), "bin": "Thùng Rác Vô Cơ"}
                )

                stream = info["stream"]
                color_rgb = info["color"]
                # OpenCV dùng BGR
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                if "Nguy Hại" in stream or "E-Waste" in stream:
                    summary["Nguy Hại"] += 1
                    has_hazard = True
                elif "Hữu Cơ" in stream:
                    summary["Hữu Cơ"] += 1
                elif "Tái Chế" in stream:
                    summary["Tái Chế"] += 1
                else:
                    summary["Khác"] += 1

                detections.append({
                    "name": cls_name,
                    "stream": stream,
                    "bin": info["bin"],
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "color": color_rgb
                })

                # Vẽ khung Bounding Box chuyên nghiệp
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color_bgr, 3)

                # Vẽ nhãn nền
                label_text = f"{stream}: {conf*100:.0f}%"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_bgr, (x1, max(0, y1 - 25)), (x1 + w + 10, y1), color_bgr, -1)
                cv2.putText(
                    annotated_bgr,
                    label_text,
                    (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        return {
            "annotated_image": annotated_pil,
            "detections": detections,
            "total_items": len(detections),
            "summary": summary,
            "has_hazard": has_hazard
        }
