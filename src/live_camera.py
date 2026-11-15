"""
live_camera.py — Ứng dụng Quét Rác Thời Gian Thực Native (Ultra Real-Time Scanner)
Kiến trúc Đa luồng Bất đồng bộ (Asynchronous Multi-threaded Pipeline):
  - Luồng Camera chính: Đọc và hiển thị khung hình ở tốc độ 30–60 FPS mượt mà tuyệt đối.
  - Luồng Worker ngầm: Suy luận YOLOv8 trên GPU RTX 2050 và cập nhật Bounding Box không gây gián đoạn video.
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
import queue
import threading
import torch
from src.detector import WasteDetector

MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"


def run_live():
    print("=" * 65)
    print("♻️  WASTECV — QUÉT RÁC THỜI GIAN THỰC ĐA LUỒNG (ZERO-LAG LIVE STREAM)")
    print("=" * 65)
    print("[*] Đang khởi động mô hình AI trên GPU...")

    detector = WasteDetector(MODEL_PATH)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[✓] Động cơ AI đang chạy trên: {device_name}")
    print("[*] Đang kết nối Webcam (Camera 0 qua DirectShow)...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] Thử kết nối camera với backend mặc định...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ LỖI: Không tìm thấy hoặc không mở được Webcam.")
        print("👉 Nếu bạn đang mở camera trên trình duyệt web, hãy bấm STOP trên web trước rồi chạy lại!")
        return

    # Cấu hình camera 640x480 tối ưu
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── HÀNG ĐỢI & LUỒNG SUY LUẬN BẤT ĐỒNG BỘ ──
    frame_queue = queue.Queue(maxsize=1)
    lock = threading.Lock()
    shared_data = {
        "detections": [],
        "has_hazard": False,
        "running": True
    }

    def ai_worker():
        while shared_data["running"]:
            try:
                frame_to_infer = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            dets, hazard = detector.get_boxes(frame_to_infer, conf_threshold=0.25, imgsz=288)
            with lock:
                shared_data["detections"] = dets
                shared_data["has_hazard"] = hazard

    worker_thread = threading.Thread(target=ai_worker, daemon=True)
    worker_thread.start()

    print("\n[✓] Camera đã bật thành công!")
    print("👉 HÃY GIƠ BẤT KỲ VẬT THỂ NÀO TRƯỚC CAMERA (Chai nhựa, cốc, vỏ lon, quả táo, điện thoại...)")
    print("👉 Bấm phím 'q' hoặc 'ESC' trên cửa sổ camera để thoát.\n")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Mất tín hiệu khung hình từ Camera.")
            break

        # Đẩy frame mới nhất vào luồng AI ngầm (không chờ đợi)
        if frame_queue.empty():
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        # Tính toán FPS thời gian thực
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - prev_time))
        prev_time = curr_time

        # Lấy kết quả AI mới nhất
        with lock:
            current_dets = list(shared_data["detections"])
            current_hazard = shared_data["has_hazard"]

        # Vẽ bounding boxes lên khung hình hiện tại (siêu nhanh < 0.2ms)
        annotated_frame = detector.draw_boxes(frame, current_dets)

        # Thanh trạng thái Header (HUD)
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, 42), (25, 25, 25), -1)
        header_text = f"WasteCV Live | FPS: {fps:.0f} | GPU: {device_name} | Phat hien: {len(current_dets)} vat the"
        cv2.putText(annotated_frame, header_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        # Cảnh báo rác nguy hại (Pin / Điện tử)
        if current_hazard:
            cv2.rectangle(annotated_frame, (0, h - 45), (w, h), (0, 0, 220), -1)
            cv2.putText(
                annotated_frame,
                "CANH BAO: PHAT HIEN PIN / RAC NGUY HAI! VUI LONG TACH RIENG!",
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow("WasteCV Station — Live Waste Detection (Nhan 'q' de thoat)", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    shared_data["running"] = False
    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Đã đóng Camera an toàn.")


if __name__ == "__main__":
    run_live()
