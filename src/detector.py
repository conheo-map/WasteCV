"""
detector.py — Hệ Thống Phát Hiện & Giám Định Đa Rác Thải Thời Gian Thực (WasteCV Advanced Detector)
Hỗ trợ 2 kiến trúc tiên tiến nhất:
  1. 🌐 YOLO-World Open-Vocabulary Engine: Nhận diện không giới hạn các loại rác đời thực phức tạp
     (từ khẩu trang, hộp xốp, ly trà sữa, vỉ thuốc đến bao bì nhiều lớp...).
  2. ⚡ YOLOv8 12 Lớp Rác Chuyên Dụng: Tối ưu hóa siêu tốc độ cho băng chuyền công nghiệp.

Tích hợp trực tiếp với IndustrialAuditor để thẩm định mã nhựa (RIC #1-#7), chỉ số ô nhiễm dầu mỡ
và ra quyết định điều phối theo Luật Bảo vệ Môi trường 2020.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO, YOLOWorld

try:
    from industrial_auditor import IndustrialAuditor, REAL_WORLD_WASTE_PROMPTS
except ImportError:
    from src.industrial_auditor import IndustrialAuditor, REAL_WORLD_WASTE_PROMPTS

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
WORLD_MODEL_PATH = MODELS_DIR / "yolov8s-worldv2.pt"
WASTE12_MODEL_PATH = MODELS_DIR / "yolov8n_waste12.pt"
COCO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"


class WasteDetector:
    def __init__(self, model_type="world", custom_prompts=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.custom_prompts = custom_prompts

        # 1. Khởi tạo mô hình theo tùy chọn
        if model_type == "world" and WORLD_MODEL_PATH.exists():
            print(f"[*] Khởi động Động cơ Open-Vocabulary YOLO-World trên {self.device}...")
            self.model = YOLOWorld(str(WORLD_MODEL_PATH))
            prompts = custom_prompts if custom_prompts else REAL_WORLD_WASTE_PROMPTS
            self.set_prompts(prompts)
            self.model_mode_desc = f"YOLO-World Open-Vocabulary ({len(self.current_prompts)} danh mục rác thực tế)"
            self.is_world = True
        elif WASTE12_MODEL_PATH.exists():
            print(f"[*] Khởi động Động cơ YOLOv8 12 Lớp Rác Chuyên Dụng trên {self.device}...")
            self.model = YOLO(str(WASTE12_MODEL_PATH))
            self.model_mode_desc = "YOLOv8 12 Lớp Rác Chuyên Dụng"
            self.is_world = False
            self.current_prompts = list(self.model.names.values())
        else:
            print(f"[*] Fallback sang YOLOv8 COCO Generic trên {self.device}...")
            self.model = YOLO(str(COCO_MODEL_PATH))
            self.model_mode_desc = "YOLOv8 Generic COCO"
            self.is_world = False
            self.current_prompts = list(self.model.names.values())

        # Warmup GPU
        if self.device == "cuda":
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=320, device="cuda", verbose=False)
        print(f"[✓] Đã sẵn sàng: {self.model_mode_desc} ({self.device})")

    def set_prompts(self, prompts: list):
        """Cập nhật danh sách prompts rác thải mới theo thời gian thực (Zero-Shot)."""
        if hasattr(self.model, "set_classes"):
            self.current_prompts = prompts
            self.model.set_classes(prompts)
            print(f"[✓] Đã nạp {len(prompts)} prompts rác thực tế vào YOLO-World.")

    def get_boxes(self, bgr_frame, conf_threshold=0.25, imgsz=320):
        """Trích xuất bounding boxes và thẩm định công nghiệp siêu tốc cho từng vật thể."""
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

                # Thẩm định công nghiệp vật liệu & độ nhiễm bẩn
                audit = IndustrialAuditor.audit_item(cls_name, conf)
                if audit["is_hazard"]:
                    has_hazard = True

                # Đổi màu HEX sang BGR để vẽ bằng OpenCV
                hex_c = audit["badge_color"].lstrip("#")
                rgb_c = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
                bgr_c = (rgb_c[2], rgb_c[1], rgb_c[0])

                detections.append({
                    "name": cls_name,
                    "vn_name": audit["vn_name"],
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "color_bgr": bgr_c,
                    "audit": audit
                })

        return detections, has_hazard

    def draw_boxes(self, bgr_frame, detections):
        """Vẽ bounding boxes thẩm định công nghiệp cực nhanh (<0.5ms)."""
        annotated_bgr = bgr_frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            color_bgr = d["color_bgr"]
            audit = d["audit"]
            conf = d["confidence"]

            # Vẽ khung chữ nhật vật thể
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color_bgr, 3)

            # Vẽ nhãn thông tin công nghiệp
            label_main = f"{audit['vn_name']} ({conf*100:.0f}%)"
            label_sub = f"{audit['decision']}"

            (w1, h1), _ = cv2.getTextSize(label_main, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            (w2, h2), _ = cv2.getTextSize(label_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            box_w = max(w1, w2) + 12
            box_h = h1 + h2 + 14

            cv2.rectangle(annotated_bgr, (x1, max(0, y1 - box_h)), (x1 + box_w, y1), color_bgr, -1)
            cv2.putText(annotated_bgr, label_main, (x1 + 6, y1 - h2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(annotated_bgr, label_sub, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return annotated_bgr

    def detect_bgr(self, bgr_frame, conf_threshold=0.25, imgsz=320):
        """Nhận diện và phân tích toàn diện trên khung hình BGR."""
        detections, has_hazard = self.get_boxes(bgr_frame, conf_threshold=conf_threshold, imgsz=imgsz)
        annotated_bgr = self.draw_boxes(bgr_frame, detections)
        summary = {"Tái Chế": 0, "Hữu Cơ": 0, "Nguy Hại": 0, "Đốt RDF / Khác": 0}

        for d in detections:
            aud = d["audit"]
            if aud["is_hazard"]:
                summary["Nguy Hại"] += 1
            elif "Hữu Cơ" in aud["target_stream"]:
                summary["Hữu Cơ"] += 1
            elif aud["is_recyclable"] and aud["contamination_score"] < 50:
                summary["Tái Chế"] += 1
            else:
                summary["Đốt RDF / Khác"] += 1

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
