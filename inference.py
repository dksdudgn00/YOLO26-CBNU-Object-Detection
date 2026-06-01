#!/usr/bin/env python3
"""
YOLO26 video inference — draws bounding boxes on every frame and saves result.
Also shows a live preview window while processing.
"""

import os
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
WORK_DIR    = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
WEIGHTS     = WORK_DIR / "runs/yolo26_train/weights/best.pt"
VIDEO_IN    = WORK_DIR / "test_video/Test Video for testing YoloV8 model.mp4"
VIDEO_OUT   = WORK_DIR / "test_video/result_inference2.mp4"
CLASSES     = ["person", "tree", "vehicle"]
CONF        = 0.25    # confidence threshold
IOU         = 0.45    # NMS IoU threshold
SHOW        = True    # show live preview (set False if no display)
# ─────────────────────────────────────────────────────────────────────────────

# Class colours  BGR
CLASS_COLORS = {
    "person":  (  0, 200,  50),   # green
    "tree":    ( 34, 139,  34),   # dark-green
    "vehicle": ( 30, 120, 255),   # blue-orange
}
DEFAULT_COLOR = (200, 200, 200)


def draw_box(frame, box, label, color, conf):
    x1, y1, x2, y2 = map(int, box)
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Label background
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_stats(frame, frame_idx, total, fps_proc, counts):
    """Top-left overlay: frame counter, processing FPS, per-class counts."""
    lines = [
        f"Frame {frame_idx}/{total}",
        f"Proc FPS: {fps_proc:.1f}",
    ] + [f"{cls}: {cnt}" for cls, cnt in counts.items()]

    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*58}")
    print(f"  YOLO26 Video Inference")
    print(f"  Weights : {WEIGHTS}")
    print(f"  Input   : {VIDEO_IN}")
    print(f"  Output  : {VIDEO_OUT}")
    print(f"  Device  : {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")
    print(f"{'='*58}\n")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}\n"
                                "Training might still be in progress.")

    model = YOLO(str(WEIGHTS))

    cap = cv2.VideoCapture(str(VIDEO_IN))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

    print(f"  {width}x{height} @ {fps:.1f}fps  |  {total} frames\n")

    frame_idx   = 0
    t_start     = time.time()
    proc_fps    = 0.0
    total_det   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Inference ────────────────────────────────────────────────────────
        t0      = time.time()
        results = model(frame, conf=CONF, iou=IOU, device=device,
                        verbose=False)[0]
        proc_fps = 1.0 / max(time.time() - t0, 1e-6)

        # ── Draw detections ───────────────────────────────────────────────────
        counts = {c: 0 for c in CLASSES}
        if results.boxes is not None and len(results.boxes):
            for box_data in results.boxes:
                xyxy  = box_data.xyxy[0].cpu().numpy()
                conf  = float(box_data.conf[0])
                cls_i = int(box_data.cls[0])
                label = CLASSES[cls_i] if cls_i < len(CLASSES) else f"cls{cls_i}"
                color = CLASS_COLORS.get(label, DEFAULT_COLOR)
                draw_box(frame, xyxy, label, color, conf)
                counts[label] = counts.get(label, 0) + 1
                total_det += 1

        draw_stats(frame, frame_idx, total, proc_fps, counts)

        out.write(frame)

        # ── Live preview ──────────────────────────────────────────────────────
        if SHOW:
            cv2.imshow("YOLO26 Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # q / ESC to quit early
                print("  Stopped early by user.")
                break

        # Progress
        if frame_idx % 30 == 0 or frame_idx == total:
            elapsed = time.time() - t_start
            eta     = (elapsed / frame_idx) * (total - frame_idx)
            print(f"  [{frame_idx:4d}/{total}]  "
                  f"proc: {proc_fps:5.1f} fps  "
                  f"det: {total_det:5d}  "
                  f"ETA: {eta:.0f}s", end="\r")

    cap.release()
    out.release()
    if SHOW:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n\n  완료!  총 {frame_idx}프레임  |  감지 수: {total_det}  |  소요: {elapsed:.1f}s")
    print(f"  결과 영상 저장 → {VIDEO_OUT}")


if __name__ == "__main__":
    main()
