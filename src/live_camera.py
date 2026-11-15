"""
live_camera.py — Ứng dụng Quét Rác Thời Gian Thực Native (Ultra Real-Time Scanner)
Chạy trực tiếp trên GPU NVIDIA RTX 2050 qua OpenCV + YOLOv8.
Tự động nhận diện đồ vật khi giơ trước webcam với tốc độ 30–60 FPS!
Bấm phím 'q' hoặc 'ESC' để thoát.
"""

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import time
import cv2
import torch
from src.detector import WasteDetector

MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"


def run_live():
    print("=" * 65)
    print("♻️  WASTECV — QUÉT RÁC THỜI GIAN THỰC QUA WEBCAM (LIVE STREAM)")
    print("=" * 65)
    print("[*] Đang khởi động mô hình AI trên GPU...")

    detector = WasteDetector(MODEL_PATH)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[✓] Động cơ AI đang chạy trên: {device_name}")
    print("[*] Đang kết nối Webcam (Camera 0)...")

    # Mở camera với DirectShow trên Windows để khởi động nhanh và tránh xung đột
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] Không mở được qua DSHOW, thử backend mặc định...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ LỖI: Không tìm thấy hoặc không mở được Webcam.")
        print("👉 Vui lòng kiểm tra quyền camera hoặc đảm bảo không có ứng dụng nào khác đang chiếm quyền camera.")
        return

    # Cấu hình độ phân giải tối ưu 640x480 để đạt 60 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[✓] Camera đã bật thành công!")
    print("👉 HÃY GIƠ BẤT KỲ VẬT THỂ NÀO TRƯỚC CAMERA (Chai nhựa, cốc, vỏ lon, quả táo, điện thoại...)")
    print("👉 Bấm phím 'q' hoặc 'ESC' trên cửa sổ camera để thoát.\n")

    prev_time = time.time()
    fps = 0
    frame_count = 0
    cached_detections = []
    has_hazard = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Mất tín hiệu khung hình từ Camera.")
            break

        frame_count += 1
        # Tính toán FPS thời gian thực
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - prev_time))
        prev_time = curr_time

        # Tối ưu hóa: Chạy suy luận YOLO mỗi 2 frame để giữ tốc độ 60 FPS mượt mà
        if frame_count % 2 == 0 or len(cached_detections) == 0:
            cached_detections, has_hazard = detector.get_boxes(frame, conf_threshold=0.25, imgsz=320)

        # Vẽ bounding boxes lên khung hình
        annotated_frame = detector.draw_boxes(frame, cached_detections)

        # Vẽ thanh trạng thái Header (HUD)
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, 45), (20, 20, 20), -1)

        # Thông tin FPS và GPU
        header_text = f"WasteCV Live | FPS: {fps:.1f} | GPU: {device_name} | Phat hien: {len(cached_detections)}"
        cv2.putText(annotated_frame, header_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Nếu phát hiện Pin / Đồ điện tử -> Hiển thị cảnh báo đỏ nhấp nháy
        if has_hazard:
            cv2.rectangle(annotated_frame, (0, h - 45), (w, h), (0, 0, 220), -1)
            cv2.putText(
                annotated_frame,
                "CANH BAO: PHAT HIEN PIN / RAC NGUY HAI! VUI LONG TACH RIENG!",
                (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow("WasteCV Station — Live Waste Detection (Nhan 'q' de thoat)", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' hoặc ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Đã đóng Camera. Chương trình kết thúc an toàn.")


if __name__ == "__main__":
    run_live()
