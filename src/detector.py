"""
detector.py — Bộ phát hiện và khoanh vùng đa vật thể rác thải thời gian thực (Real-time YOLOv8 Waste Detector)
Tích hợp 2 chế độ:
  1. YOLOv8 12 Lớp Rác Chuyên Dụng (battery, biological, brown-glass, cardboard, clothes,
     green-glass, metal, paper, plastic, shoes, trash, white-glass)
  2. Fallback sang COCO 80 lớp khi cần nhận diện các vật dụng văn phòng / sinh hoạt thường ngày.

Phân luồng 4 nhóm rác thực tế:
  - 🟢 Xanh lá: Rác Tái Chế (Nhựa, Thủy tinh, Kim loại, Giấy, Carton)
  - 🟠 Cam:    Rác Hữu Cơ / Phân Hủy Sinh Học (Đồ ăn, Rau củ, Vỏ trái cây)
  - 🔴 Đỏ:     Rác Nguy Hại / Pin / E-Waste (Pin, Đồ điện tử rò rỉ hóa chất)
  - 🟣 Tím:    Đồ Cũ Tái Sử Dụng (Quần áo, Giày dép)
  - ⚪ Xám:    Rác Khác / Vô Cơ (Rác chôn lấp, túi bẩn)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8n_waste12.pt"
if not DEFAULT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

# ── BẢN ĐỒ 12 LỚP RÁC CHUYÊN SÂU ──
WASTE_12_CLASS_MAP = {
    "battery": {
        "stream": "Nguy Hại (Pin / E-Waste)",
        "vn_name": "Pin & Rác Nguy Hại",
        "bin": "Thùng Rác Nguy Hại (Màu Đỏ)",
        "color_rgb": (231, 76, 60),      # Đỏ
        "category": "Nguy Hại",
        "is_hazard": True
    },
    "biological": {
        "stream": "Hữu Cơ (Thực phẩm/Rau củ)",
        "vn_name": "Rác Hữu Cơ Sinh Học",
        "bin": "Thùng Rác Hữu Cơ (Ủ Compost)",
        "color_rgb": (230, 126, 34),     # Cam
        "category": "Hữu Cơ",
        "is_hazard": False
    },
    "brown-glass": {
        "stream": "Tái Chế (Thủy tinh Nâu)",
        "vn_name": "Thủy Tinh Nâu (Chai bia, siro)",
        "bin": "Thùng Tái Chế Thủy Tinh",
        "color_rgb": (46, 204, 113),     # Xanh lá
        "category": "Tái Chế",
        "is_hazard": False
    },
    "green-glass": {
        "stream": "Tái Chế (Thủy tinh Xanh)",
        "vn_name": "Thủy Tinh Xanh Lục",
        "bin": "Thùng Tái Chế Thủy Tinh",
        "color_rgb": (46, 204, 113),
        "category": "Tái Chế",
        "is_hazard": False
    },
    "white-glass": {
        "stream": "Tái Chế (Thủy tinh Trong)",
        "vn_name": "Thủy Tinh Trong Suốt",
        "bin": "Thùng Tái Chế Thủy Tinh",
        "color_rgb": (26, 188, 156),     # Ngọc lam
        "category": "Tái Chế",
        "is_hazard": False
    },
    "cardboard": {
        "stream": "Tái Chế (Bìa Carton)",
        "vn_name": "Hộp & Bìa Carton",
        "bin": "Thùng Tái Chế Giấy",
        "color_rgb": (52, 152, 219),     # Xanh dương
        "category": "Tái Chế",
        "is_hazard": False
    },
    "paper": {
        "stream": "Tái Chế (Giấy Báo)",
        "vn_name": "Giấy Vụn & Báo Cũ",
        "bin": "Thùng Tái Chế Giấy",
        "color_rgb": (52, 152, 219),
        "category": "Tái Chế",
        "is_hazard": False
    },
    "metal": {
        "stream": "Tái Chế (Kim Loại/Lon)",
        "vn_name": "Vỏ Lon & Kim Loại",
        "bin": "Thùng Tái Chế Kim Loại",
        "color_rgb": (46, 204, 113),
        "category": "Tái Chế",
        "is_hazard": False
    },
    "plastic": {
        "stream": "Tái Chế (Đồ Nhựa)",
        "vn_name": "Chai & Hộp Nhựa",
        "bin": "Thùng Tái Chế Nhựa",
        "color_rgb": (241, 196, 15),     # Vàng
        "category": "Tái Chế",
        "is_hazard": False
    },
    "clothes": {
        "stream": "Tái Chế / Đồ Cũ (Vải)",
        "vn_name": "Quần Áo Cũ & Vải",
        "bin": "Thùng Thu Gom Vải / Đồ Cũ",
        "color_rgb": (155, 89, 182),     # Tím
        "category": "Tái Chế",
        "is_hazard": False
    },
    "shoes": {
        "stream": "Đồ Cũ (Giày Dép)",
        "vn_name": "Giày Dép Cũ",
        "bin": "Thùng Thu Gom Đồ Dùng Cũ",
        "color_rgb": (142, 68, 173),     # Tím đậm
        "category": "Tái Chế",
        "is_hazard": False
    },
    "trash": {
        "stream": "Rác Vô Cơ Khác",
        "vn_name": "Rác Vô Cơ Chôn Lấp",
        "bin": "Thùng Rác Vô Cơ / Khác",
        "color_rgb": (149, 165, 166),    # Xám
        "category": "Khác",
        "is_hazard": False
    }
}

# ── BẢN ĐỒ COCO FALLBACK ──
COCO_FALLBACK_MAP = {
    "bottle": {"stream": "Tái Chế (Chai lọ)", "vn_name": "Chai nước", "bin": "Thùng Tái Chế", "color_rgb": (46, 204, 113), "category": "Tái Chế", "is_hazard": False},
    "cup": {"stream": "Tái Chế (Ly / Cốc)", "vn_name": "Ly cốc nhựa/thủy tinh", "bin": "Thùng Tái Chế", "color_rgb": (46, 204, 113), "category": "Tái Chế", "is_hazard": False},
    "wine glass": {"stream": "Tái Chế (Thủy Tinh)", "vn_name": "Ly thủy tinh", "bin": "Thùng Tái Chế", "color_rgb": (46, 204, 113), "category": "Tái Chế", "is_hazard": False},
    "bowl": {"stream": "Tái Chế (Bát / Hộp)", "vn_name": "Bát hộp nhựa", "bin": "Thùng Tái Chế", "color_rgb": (46, 204, 113), "category": "Tái Chế", "is_hazard": False},
    "book": {"stream": "Tái Chế (Sách báo / Giấy)", "vn_name": "Sách báo giấy", "bin": "Thùng Tái Chế Giấy", "color_rgb": (52, 152, 219), "category": "Tái Chế", "is_hazard": False},
    "banana": {"stream": "Hữu Cơ (Vỏ chuối)", "vn_name": "Vỏ chuối / Trái cây", "bin": "Thùng Rác Hữu Cơ", "color_rgb": (230, 126, 34), "category": "Hữu Cơ", "is_hazard": False},
    "apple": {"stream": "Hữu Cơ (Trái táo)", "vn_name": "Vỏ táo / Trái cây", "bin": "Thùng Rác Hữu Cơ", "color_rgb": (230, 126, 34), "category": "Hữu Cơ", "is_hazard": False},
    "orange": {"stream": "Hữu Cơ (Vỏ cam)", "vn_name": "Vỏ cam / Trái cây", "bin": "Thùng Rác Hữu Cơ", "color_rgb": (230, 126, 34), "category": "Hữu Cơ", "is_hazard": False},
    "sandwich": {"stream": "Hữu Cơ (Thức ăn thừa)", "vn_name": "Bánh mì / Đồ ăn thừa", "bin": "Thùng Rác Hữu Cơ", "color_rgb": (230, 126, 34), "category": "Hữu Cơ", "is_hazard": False},
    "cell phone": {"stream": "Nguy Hại (Điện thoại/Pin)", "vn_name": "Điện thoại / E-Waste", "bin": "Thùng Rác Nguy Hại", "color_rgb": (231, 76, 60), "category": "Nguy Hại", "is_hazard": True},
    "mouse": {"stream": "E-Waste (Chuột máy tính)", "vn_name": "Chuột máy tính cũ", "bin": "Thùng Rác Điện Tử", "color_rgb": (231, 76, 60), "category": "Nguy Hại", "is_hazard": True},
    "keyboard": {"stream": "E-Waste (Bàn phím)", "vn_name": "Bàn phím cũ", "bin": "Thùng Rác Điện Tử", "color_rgb": (231, 76, 60), "category": "Nguy Hại", "is_hazard": True},
    "backpack": {"stream": "Đồ Cũ / Vải (Balo)", "vn_name": "Balo cũ", "bin": "Thùng Đồ Cũ / Vải", "color_rgb": (155, 89, 182), "category": "Tái Chế", "is_hazard": False},
}


class WasteDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.model_path))

        # Kiểm tra xem mô hình hiện tại là 12 lớp rác thải hay COCO
        model_classes = self.model.names
        self.is_12class_model = "battery" in model_classes.values() or 0 in model_classes and model_classes[0] == "battery"
        self.model_mode_desc = "YOLOv8 12 Lớp Rác Chuyên Dụng" if self.is_12class_model else "YOLOv8 Generic COCO"

        # Warmup GPU
        if self.device == "cuda":
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=320, device="cuda", verbose=False)
        print(f"[✓] Đã khởi tạo WasteDetector ({self.model_mode_desc}) trên thiết bị: {self.device}")

    def _resolve_info(self, cls_name):
        """Ánh xạ nhãn model sang thông tin phân loại rác chi tiết."""
        if self.is_12class_model and cls_name in WASTE_12_CLASS_MAP:
            return WASTE_12_CLASS_MAP[cls_name]
        if cls_name in COCO_FALLBACK_MAP:
            return COCO_FALLBACK_MAP[cls_name]
        return {
            "stream": f"Vật thể ({cls_name})",
            "vn_name": cls_name,
            "bin": "Thùng Rác Vô Cơ",
            "color_rgb": (149, 165, 166),
            "category": "Khác",
            "is_hazard": False
        }

    def get_boxes(self, bgr_frame, conf_threshold=0.25, imgsz=320):
        """Trích xuất bounding boxes siêu tốc (cho async queue worker)."""
        results = self.model.predict(
            source=bgr_frame,
            conf=conf_threshold,
            device=self.device,
            imgsz=imgsz,
            verbose=False
        )
        detections = []
        has_hazard = False
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = results[0].names[cls_id]
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                info = self._resolve_info(cls_name)
                color_rgb = info["color_rgb"]
                if info["is_hazard"]:
                    has_hazard = True

                detections.append({
                    "name": cls_name,
                    "vn_name": info["vn_name"],
                    "stream": info["stream"],
                    "bin": info["bin"],
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "color_bgr": (color_rgb[2], color_rgb[1], color_rgb[0])
                })
        return detections, has_hazard

    def draw_boxes(self, bgr_frame, detections):
        """Vẽ bounding boxes cực nhanh (<0.5ms) lên khung hình hiện tại."""
        annotated_bgr = bgr_frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            color_bgr = d["color_bgr"]
            stream = d["stream"]
            conf = d["confidence"]

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color_bgr, 3)
            label_text = f"{stream}: {conf*100:.0f}%"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_bgr, (x1, max(0, y1 - 25)), (x1 + w + 10, y1), color_bgr, -1)
            cv2.putText(annotated_bgr, label_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return annotated_bgr

    def detect_bgr(self, bgr_frame, conf_threshold=0.25, imgsz=320):
        """Nhận diện đồng thời phân tích thống kê trên khung hình BGR."""
        detections, has_hazard = self.get_boxes(bgr_frame, conf_threshold=conf_threshold, imgsz=imgsz)
        annotated_bgr = self.draw_boxes(bgr_frame, detections)
        summary = {"Tái Chế": 0, "Hữu Cơ": 0, "Nguy Hại": 0, "Khác": 0}
        for d in detections:
            cat = self._resolve_info(d["name"])["category"]
            if cat in summary:
                summary[cat] += 1
            else:
                summary["Khác"] += 1
        return annotated_bgr, detections, summary, has_hazard

    def detect(self, image, conf_threshold=0.25, imgsz=640):
        """Nhận diện ảnh tĩnh PIL hoặc numpy RGB với độ phân giải cao."""
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        else:
            img_rgb = image.copy()

        bgr_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr, detections, summary, has_hazard = self.detect_bgr(
            bgr_frame, conf_threshold=conf_threshold, imgsz=imgsz
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return {
            "annotated_image": Image.fromarray(annotated_rgb),
            "detections": detections,
            "total_items": len(detections),
            "summary": summary,
            "has_hazard": has_hazard
        }
