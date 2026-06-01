#!/usr/bin/env python3
"""
Fine-tuned vs Pre-trained  —  Side-by-side 비교 영상 생성
각 프레임을 절반 크기로 좌(fine-tuned) / 우(pre-trained) 배치 → mp4 저장
"""

import cv2
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
FINETUNED_W   = WORK_DIR / "runs/yolo26_train/weights/best.pt"
PRETRAINED_W  = WORK_DIR / "yolo26n.pt"
INPUT_DIR     = WORK_DIR / "datasets/no_annotation/origin_images"
SAVE_DIR      = WORK_DIR / "comparison_results"
BAG           = "bag2"        # bag1 / bag2 / bag3  (비교할 영상)
VIDEO_FPS     = 30.0
OUTPUT_W      = 1920          # 출력 영상 너비 (각 모델 960px씩)
OUTPUT_H      = 580           # 각 프레임 540px + 상단 헤더 40px
CONF          = 0.25
IOU           = 0.45
# ──────────────────────────────────────────────────────────────────────────────

FRAME_W = OUTPUT_W // 2       # 한 쪽 프레임 너비 = 960
FRAME_H = OUTPUT_H - 40       # 헤더 제외 프레임 높이 = 540

# 클래스 색상 (BGR)
COLORS_FT = {0: (0, 200, 50),  2: (30, 120, 255)}   # fine-tuned
COLORS_PT = {0: (0, 200, 50),  2: (30, 120, 255)}   # pre-trained (같은 색)
CLASS_NAMES_FT = {0: "person", 1: "tree", 2: "vehicle"}

# COCO → 우리 클래스 매핑 (pre-trained용)
COCO_TO_OURS = {0: 0, 1: 2, 2: 2, 3: 2, 5: 2, 7: 2}
CLASS_NAMES_PT = {0: "person", 2: "vehicle"}


def draw_boxes(img, boxes_data, class_names, colors):
    """boxes_data: list of (xyxy, cls_id, conf)"""
    for xyxy, cls_id, conf in boxes_data:
        x1, y1, x2, y2 = map(int, xyxy)
        color = colors.get(cls_id, (200, 200, 200))
        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_info(img, title, n_person, n_vehicle, fps, color):
    """좌상단 정보 오버레이"""
    lines = [
        f"Person:  {n_person}",
        f"Vehicle: {n_vehicle}",
        f"FPS: {fps:.1f}",
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


def make_header(frame_idx, total):
    """상단 헤더 바 (40px)"""
    header = np.zeros((40, OUTPUT_W, 3), dtype=np.uint8)
    header[:] = (40, 40, 40)

    # 구분선
    cv2.line(header, (OUTPUT_W // 2, 0), (OUTPUT_W // 2, 40), (80, 80, 80), 1)

    # 제목
    cv2.putText(header, "Fine-tuned (best.pt)",
                (OUTPUT_W // 4 - 120, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(header, "Pre-trained (yolo26n.pt)",
                (OUTPUT_W * 3 // 4 - 140, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 180, 255), 2, cv2.LINE_AA)

    # 프레임 카운터
    counter = f"Frame {frame_idx}/{total}"
    (tw, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(header, counter,
                (OUTPUT_W // 2 - tw // 2, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return header


def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Side-by-Side 비교 영상 생성")
    print(f"  Fine-tuned : {FINETUNED_W.name}")
    print(f"  Pre-trained: {PRETRAINED_W.name}")
    print(f"  Bag        : {BAG}")
    print(f"  출력 해상도 : {OUTPUT_W}x{OUTPUT_H}  (@{VIDEO_FPS}fps)")
    print(f"{'='*60}\n")

    # 모델 로드
    print("  모델 로드 중...")
    model_ft = YOLO(str(FINETUNED_W))
    model_pt = YOLO(str(PRETRAINED_W))

    # 이미지 목록 (프레임 번호 순 정렬)
    img_dir = INPUT_DIR / BAG
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        (p for p in img_dir.iterdir() if p.suffix.lower() in exts),
        key=lambda p: p.stem
    )
    total = len(images)
    print(f"  프레임 수 : {total}\n")

    # VideoWriter
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / f"compare_{BAG}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, VIDEO_FPS, (OUTPUT_W, OUTPUT_H))

    t_start = time.time()

    for idx, img_path in enumerate(images, 1):
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue

        # ── Fine-tuned 추론 ────────────────────────────────────────────
        t0 = time.time()
        res_ft = model_ft(orig, conf=CONF, iou=IOU, device=device, verbose=False)[0]
        fps_ft = 1.0 / max(time.time() - t0, 1e-6)

        det_ft, np_ft, nv_ft = [], 0, 0
        if res_ft.boxes is not None:
            for box in res_ft.boxes:
                cid = int(box.cls[0])
                cf  = float(box.conf[0])
                det_ft.append((box.xyxy[0].cpu().numpy(), cid, cf))
                if cid == 0: np_ft += 1
                elif cid == 2: nv_ft += 1

        # ── Pre-trained 추론 ───────────────────────────────────────────
        t0 = time.time()
        res_pt = model_pt(orig, conf=CONF, iou=IOU, device=device,
                          verbose=False, classes=list(COCO_TO_OURS.keys()))[0]
        fps_pt = 1.0 / max(time.time() - t0, 1e-6)

        det_pt, np_pt, nv_pt = [], 0, 0
        if res_pt.boxes is not None:
            for box in res_pt.boxes:
                cid_coco = int(box.cls[0])
                cid = COCO_TO_OURS.get(cid_coco)
                if cid is None: continue
                cf  = float(box.conf[0])
                det_pt.append((box.xyxy[0].cpu().numpy(), cid, cf))
                if cid == 0: np_pt += 1
                elif cid == 2: nv_pt += 1

        # ── 프레임 합성 ────────────────────────────────────────────────
        # 각 프레임 절반 크기로 리사이즈
        frame_ft = cv2.resize(orig.copy(), (FRAME_W, FRAME_H))
        frame_pt = cv2.resize(orig.copy(), (FRAME_W, FRAME_H))

        # 리사이즈 비율에 맞게 bbox 좌표 변환
        sx = FRAME_W / orig.shape[1]
        sy = FRAME_H / orig.shape[0]

        def scale_det(det_list):
            scaled = []
            for (xyxy, cid, cf) in det_list:
                x1, y1, x2, y2 = xyxy
                scaled.append((
                    [x1*sx, y1*sy, x2*sx, y2*sy], cid, cf
                ))
            return scaled

        draw_boxes(frame_ft, scale_det(det_ft), CLASS_NAMES_FT, COLORS_FT)
        draw_boxes(frame_pt, scale_det(det_pt), CLASS_NAMES_PT, COLORS_PT)

        draw_info(frame_ft, "Fine-tuned",  np_ft, nv_ft, fps_ft, (100, 220, 100))
        draw_info(frame_pt, "Pre-trained", np_pt, nv_pt, fps_pt, (100, 180, 255))

        # 구분선
        cv2.line(frame_ft, (FRAME_W - 1, 0), (FRAME_W - 1, FRAME_H), (80, 80, 80), 2)

        # 헤더 + 두 프레임 합치기
        header  = make_header(idx, total)
        body    = np.hstack([frame_ft, frame_pt])
        canvas  = np.vstack([header, body])

        writer.write(canvas)

        # 진행률
        if idx % 30 == 0 or idx == total:
            elapsed = time.time() - t_start
            eta = (elapsed / idx) * (total - idx)
            print(f"  [{idx:4d}/{total}]  elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s",
                  end="\r")

    writer.release()
    print(f"\n\n  완료! → {out_path}")


if __name__ == "__main__":
    main()
