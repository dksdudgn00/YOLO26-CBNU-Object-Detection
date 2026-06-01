#!/usr/bin/env python3
"""
Inference on no_annotation images using pre-trained yolo26n.pt (COCO weights).
Detects only: person + vehicle (car/bus/truck/motorcycle/bicycle)
No ground-truth labels → reports detection statistics & confidence distribution.
"""

import os
import json
import time
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────
WORK_DIR    = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
WEIGHTS     = WORK_DIR / "yolo26n.pt"
INPUT_DIR   = WORK_DIR / "datasets/no_annotation/origin_images"
SAVE_DIR    = WORK_DIR / "pretrained_inference_results"
CONF        = 0.25
IOU         = 0.45
SAVE_IMGS   = False  # annotated 이미지 저장 여부
SAVE_LABELS = True   # YOLO 형식 라벨 .txt 저장 (pseudo-labeling)
SAVE_VIDEO  = True   # bag별 추론 결과를 mp4 영상으로 저장
VIDEO_FPS   = 30.0   # 출력 영상 FPS
# ─────────────────────────────────────────────────────────────────────────────

# COCO class ID → 우리 데이터셋 class ID 매핑
# 우리 classes: 0=person, 1=tree, 2=vehicle
COCO_TO_OUR_CLASS = {
    0: 0,   # person  → person(0)
    1: 2,   # bicycle → vehicle(2)
    2: 2,   # car     → vehicle(2)
    3: 2,   # motorcycle → vehicle(2)
    5: 2,   # bus     → vehicle(2)
    7: 2,   # truck   → vehicle(2)
}

HAS_DISPLAY = bool(os.environ.get("DISPLAY", ""))
matplotlib.use("TkAgg" if HAS_DISPLAY else "Agg")

# COCO class ID → 우리 레이블 매핑 (person / vehicle 만)
COCO_TARGET = {
    0: "person",
    1: "vehicle",   # bicycle
    2: "vehicle",   # car
    3: "vehicle",   # motorcycle
    5: "vehicle",   # bus
    7: "vehicle",   # truck
}
COLORS = {"person": (0, 200, 50), "vehicle": (30, 120, 255)}


def draw_box(img, xyxy, label, conf, color):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run():
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Pre-trained yolo26n  —  Person & Vehicle Detection")
    print(f"  Weights : {WEIGHTS}")
    print(f"  Input   : {INPUT_DIR}")
    print(f"  Device  : {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")
    print(f"{'='*60}\n")

    model = YOLO(str(WEIGHTS))

    # 모든 이미지 수집 (bag1/2/3)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted(
        p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in exts
    )
    total = len(all_images)
    print(f"  총 이미지: {total}장  (bag1/2/3)\n")

    # 저장 폴더 생성
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for bag in ["bag1", "bag2", "bag3"]:
        if SAVE_IMGS:
            (SAVE_DIR / "images" / bag).mkdir(parents=True, exist_ok=True)
        if SAVE_LABELS:
            (SAVE_DIR / "labels" / bag).mkdir(parents=True, exist_ok=True)
    if SAVE_VIDEO:
        (SAVE_DIR / "videos").mkdir(parents=True, exist_ok=True)

    # ── 통계 컨테이너 ─────────────────────────────────────────────────────
    stats = defaultdict(lambda: {
        "total_images": 0,
        "images_with_det": 0,
        "det_counts": {"person": 0, "vehicle": 0},
        "confs": {"person": [], "vehicle": []},
    })

    # bag별로 분리해서 순서대로 처리 (프레임 번호 순 정렬)
    bags_images = defaultdict(list)
    for p in all_images:
        bags_images[p.parent.name].append(p)
    for bag in bags_images:
        bags_images[bag].sort(key=lambda p: p.stem)  # 000000, 000001, ... 순서

    t_start   = time.time()
    label_names = {0: "person", 2: "vehicle"}
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    idx       = 0

    for bag in sorted(bags_images.keys()):
        images = bags_images[bag]

        # VideoWriter 초기화 (첫 프레임 크기 기준)
        writer = None
        if SAVE_VIDEO:
            sample = cv2.imread(str(images[0]))
            fh, fw = sample.shape[:2]
            video_path = SAVE_DIR / "videos" / f"{bag}_inference.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (fw, fh))
            print(f"  [{bag}] 영상 생성 중: {video_path.name}  ({len(images)} frames)")

        for img_path in images:
            idx += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            results = model(img, conf=CONF, iou=IOU, device=device,
                            verbose=False, classes=list(COCO_TO_OUR_CLASS.keys()))[0]

            s = stats[bag]
            s["total_images"] += 1
            has_det = False
            yolo_lines = []

            if results.boxes is not None and len(results.boxes):
                has_det = True
                for box in results.boxes:
                    cid_coco = int(box.cls[0])
                    our_cid  = COCO_TO_OUR_CLASS.get(cid_coco)
                    if our_cid is None:
                        continue
                    cf   = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = xyxy
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    yolo_lines.append(f"{our_cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    label = label_names[our_cid]
                    s["det_counts"][label] += 1
                    s["confs"][label].append(cf)
                    draw_box(img, xyxy, label, cf, COLORS[label])

            if has_det:
                s["images_with_det"] += 1

            if SAVE_LABELS:
                lbl_path = SAVE_DIR / "labels" / bag / (img_path.stem + ".txt")
                lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            if SAVE_IMGS and idx % 100 == 0:
                out_path = SAVE_DIR / "images" / bag / img_path.name
                cv2.imwrite(str(out_path), img)

            if SAVE_VIDEO and writer:
                writer.write(img)

            # 진행률
            if idx % 100 == 0 or idx == total:
                elapsed = time.time() - t_start
                eta = (elapsed / idx) * (total - idx)
                print(f"  [{idx:4d}/{total}]  {bag}  elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s",
                      end="\r")

        if writer:
            writer.release()
            print(f"\n  저장 완료 → {video_path}")

    print(f"\n  추론 완료  |  소요: {time.time() - t_start:.1f}s")
    return stats


def print_and_save_stats(stats):
    print(f"\n{'─'*60}")
    print(f"  {'Bag':<6}  {'Images':>7}  {'w/ Det':>7}  "
          f"{'Person':>8}  {'Vehicle':>9}  {'P conf':>7}  {'V conf':>7}")
    print(f"{'─'*60}")

    summary = {"bags": {}}
    all_p_confs, all_v_confs = [], []

    for bag in sorted(stats.keys()):
        s = stats[bag]
        n        = s["total_images"]
        n_det    = s["images_with_det"]
        n_p      = s["det_counts"]["person"]
        n_v      = s["det_counts"]["vehicle"]
        p_confs  = s["confs"]["person"]
        v_confs  = s["confs"]["vehicle"]
        avg_pc   = np.mean(p_confs) if p_confs else 0.0
        avg_vc   = np.mean(v_confs) if v_confs else 0.0

        print(f"  {bag:<6}  {n:>7}  {n_det:>7}  "
              f"{n_p:>8}  {n_v:>9}  {avg_pc:>7.3f}  {avg_vc:>7.3f}")

        summary["bags"][bag] = {
            "total_images": n, "images_with_detection": n_det,
            "person_detections": n_p, "vehicle_detections": n_v,
            "avg_conf_person": round(avg_pc, 4),
            "avg_conf_vehicle": round(avg_vc, 4),
        }
        all_p_confs += p_confs
        all_v_confs += v_confs

    # 전체 합산
    total_imgs = sum(s["total_images"]    for s in stats.values())
    total_det  = sum(s["images_with_det"] for s in stats.values())
    total_p    = sum(s["det_counts"]["person"]  for s in stats.values())
    total_v    = sum(s["det_counts"]["vehicle"] for s in stats.values())
    avg_pc_all = np.mean(all_p_confs) if all_p_confs else 0.0
    avg_vc_all = np.mean(all_v_confs) if all_v_confs else 0.0

    print(f"{'─'*60}")
    print(f"  {'ALL':<6}  {total_imgs:>7}  {total_det:>7}  "
          f"{total_p:>8}  {total_v:>9}  {avg_pc_all:>7.3f}  {avg_vc_all:>7.3f}")
    print(f"{'─'*60}\n")

    print("  ※ mAP는 ground-truth 라벨이 없어 계산 불가합니다.")
    print("    대신 confidence 분포로 모델 신뢰도를 확인하세요.\n")

    summary["total"] = {
        "total_images": total_imgs, "images_with_detection": total_det,
        "person_detections": total_p, "vehicle_detections": total_v,
        "avg_conf_person": round(avg_pc_all, 4),
        "avg_conf_vehicle": round(avg_vc_all, 4),
    }

    out = SAVE_DIR / "stats.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  통계 저장 → {out}")

    return all_p_confs, all_v_confs


def plot_confidence(all_p_confs, all_v_confs, stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Pre-trained yolo26n  —  Detection Statistics", fontsize=13, fontweight="bold")

    # ① Confidence 히스토그램
    ax = axes[0]
    bins = np.linspace(0.25, 1.0, 20)
    if all_p_confs:
        ax.hist(all_p_confs, bins=bins, alpha=0.7, color="#2196F3", label=f"person (n={len(all_p_confs)})")
    if all_v_confs:
        ax.hist(all_v_confs, bins=bins, alpha=0.7, color="#FF5722", label=f"vehicle (n={len(all_v_confs)})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # ② bag별 감지 수 막대그래프
    ax = axes[1]
    bags = sorted(stats.keys())
    x = np.arange(len(bags))
    w = 0.35
    p_counts = [stats[b]["det_counts"]["person"]  for b in bags]
    v_counts = [stats[b]["det_counts"]["vehicle"] for b in bags]
    ax.bar(x - w/2, p_counts, w, label="person",  color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, v_counts, w, label="vehicle", color="#FF5722", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(bags)
    ax.set_ylabel("Detections")
    ax.set_title("Detections per Bag")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ③ 이미지당 탐지 비율 (파이 차트)
    ax = axes[2]
    total_imgs = sum(s["total_images"]    for s in stats.values())
    total_det  = sum(s["images_with_det"] for s in stats.values())
    no_det     = total_imgs - total_det
    ax.pie([total_det, no_det],
           labels=[f"감지 있음\n({total_det})", f"감지 없음\n({no_det})"],
           colors=["#4CAF50", "#9E9E9E"], autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("Detection Rate")

    plt.tight_layout()
    out = SAVE_DIR / "detection_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  통계 차트 저장 → {out}")
    if HAS_DISPLAY:
        plt.show()
    plt.close(fig)


def main():
    stats = run()
    all_p_confs, all_v_confs = print_and_save_stats(stats)
    plot_confidence(all_p_confs, all_v_confs, stats)
    print(f"\n  결과 저장 위치: {SAVE_DIR}/")
    if SAVE_IMGS:
        print(f"  annotated 이미지: {SAVE_DIR}/images/bag*/")
    if SAVE_LABELS:
        print(f"  YOLO 라벨 (.txt): {SAVE_DIR}/labels/bag*/")
        print(f"  → 이 라벨로 추가 학습 가능 (class 0=person, 2=vehicle)")
    if SAVE_VIDEO:
        print(f"  추론 영상 (.mp4): {SAVE_DIR}/videos/bag*_inference.mp4")


if __name__ == "__main__":
    main()
