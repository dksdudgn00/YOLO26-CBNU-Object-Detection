#!/usr/bin/env python3
"""
Fine-tuned (best.pt) vs Pre-trained (yolo26n.pt) 성능 비교
- 동일한 val 세트 + 실제 ground-truth 라벨 사용
- person / vehicle 클래스 비교 (tree는 COCO에 없어 제외)
- mAP@0.5, mAP@0.5:0.95, Precision, Recall 출력 + 차트 저장
"""

import os
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR       = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
FINETUNED_W    = WORK_DIR / "runs/yolo26_train/weights/best.pt"
PRETRAINED_W   = WORK_DIR / "yolo26n.pt"
VAL_TXT        = WORK_DIR / "datasets/val.txt"
LABELS_DIR     = WORK_DIR / "datasets/detect/labels"
SAVE_DIR       = WORK_DIR / "comparison_results"
CONF           = 0.5
IOU_NMS        = 0.45
# 비교 대상 클래스 (tree 제외 — COCO에 없음)
COMPARE_CLASSES = {0: "person", 2: "vehicle"}

# COCO ID → 우리 클래스 ID 매핑 (pre-trained용)
COCO_TO_OURS = {0: 0, 1: 2, 2: 2, 3: 2, 5: 2, 7: 2}
# ──────────────────────────────────────────────────────────────────────────────

HAS_DISPLAY = bool(os.environ.get("DISPLAY", ""))
matplotlib.use("TkAgg" if HAS_DISPLAY else "Agg")


def load_val_data(val_txt: Path, labels_dir: Path):
    """val.txt에서 이미지 경로와 대응하는 GT 라벨 로드."""
    img_paths = [Path(p.strip()) for p in val_txt.read_text().splitlines() if p.strip()]
    data = []
    for img_path in img_paths:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        boxes, labels = [], []
        for line in lbl_path.read_text().splitlines():
            f = line.split()
            if len(f) != 5:
                continue
            cls, cx, cy, w, h = int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4])
            if cls not in COMPARE_CLASSES:   # person / vehicle 만
                continue
            boxes.append([cx, cy, w, h])
            labels.append(cls)
        if boxes:
            data.append({"img": img_path, "boxes": boxes, "labels": labels})
    return data


def xywh_to_xyxy(boxes, w_img, h_img):
    """정규화 cx cy w h → 픽셀 x1 y1 x2 y2"""
    result = []
    for cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2) * w_img
        y1 = (cy - bh / 2) * h_img
        x2 = (cx + bw / 2) * w_img
        y2 = (cy + bh / 2) * h_img
        result.append([x1, y1, x2, y2])
    return result


def run_eval(model_path: Path, val_data: list, coco_remap: bool = False):
    """모델 추론 후 torchmetrics MeanAveragePrecision으로 mAP 계산."""
    from ultralytics import YOLO
    import cv2

    device = 0 if torch.cuda.is_available() else "cpu"
    model  = YOLO(str(model_path))
    coco_classes = list(COCO_TO_OURS.keys()) if coco_remap else None

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    total  = len(val_data)

    for idx, sample in enumerate(val_data, 1):
        img_bgr = cv2.imread(str(sample["img"]))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # GT
        gt_xyxy = xywh_to_xyxy(sample["boxes"], w, h)
        gt_dict = {
            "boxes":  torch.tensor(gt_xyxy,         dtype=torch.float32),
            "labels": torch.tensor(sample["labels"], dtype=torch.int64),
        }

        # 예측
        results = model(img_bgr, conf=CONF, iou=IOU_NMS,
                        device=device, verbose=False,
                        classes=coco_classes)[0]

        pred_boxes, pred_labels, pred_scores = [], [], []
        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                cid_raw = int(box.cls[0])
                cid     = COCO_TO_OURS.get(cid_raw) if coco_remap else cid_raw
                if cid not in COMPARE_CLASSES:
                    continue
                pred_boxes.append(box.xyxy[0].cpu().tolist())
                pred_labels.append(cid)
                pred_scores.append(float(box.conf[0]))

        pred_dict = {
            "boxes":  torch.tensor(pred_boxes,  dtype=torch.float32) if pred_boxes
                      else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(pred_labels, dtype=torch.int64),
            "scores": torch.tensor(pred_scores, dtype=torch.float32),
        }

        metric.update([pred_dict], [gt_dict])

        if idx % 20 == 0 or idx == total:
            print(f"    [{idx:3d}/{total}]", end="\r")

    print()
    return metric.compute()


def extract_class_metrics(result, label_map):
    """torchmetrics 결과에서 클래스별 수치 추출."""
    classes = {}
    if "map_per_class" in result and result["map_per_class"] is not None:
        maps = result["map_per_class"].tolist()
        for i, (cid, cname) in enumerate(label_map.items()):
            if i < len(maps):
                classes[cname] = {"mAP50_95": round(maps[i], 4)}
    if "mar_100_per_class" in result:
        pass  # recall per class available if needed
    return classes


def print_comparison(res_ft, res_pt, name_ft="fine-tuned", name_pt="pre-trained"):
    keys = ["map_50", "map_75", "map", "mar_100"]
    labels_disp = {
        "map_50":   "mAP@0.5",
        "map_75":   "mAP@0.75",
        "map":      "mAP@0.5:0.95",
        "mar_100":  "Recall@100",
    }
    print(f"\n{'═'*52}")
    print(f"  {'Metric':<16}  {name_ft:>14}  {name_pt:>14}")
    print(f"{'─'*52}")
    rows = {}
    for k in keys:
        ft_v = float(res_ft.get(k, 0))
        pt_v = float(res_pt.get(k, 0))
        diff = ft_v - pt_v
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
        label = labels_disp.get(k, k)
        print(f"  {label:<16}  {ft_v:>14.4f}  {pt_v:>14.4f}  {arrow}{abs(diff):.4f}")
        rows[label] = (ft_v, pt_v)
    print(f"{'─'*52}")

    # 클래스별 mAP
    if "map_per_class" in res_ft and res_ft["map_per_class"] is not None:
        print(f"\n  클래스별 mAP@0.5:0.95")
        print(f"  {'Class':<12}  {name_ft:>14}  {name_pt:>14}")
        print(f"  {'─'*44}")
        ft_per = res_ft["map_per_class"].tolist()
        pt_per = res_pt["map_per_class"].tolist()
        for i, (cid, cname) in enumerate(COMPARE_CLASSES.items()):
            ft_v = ft_per[i] if i < len(ft_per) else 0.0
            pt_v = pt_per[i] if i < len(pt_per) else 0.0
            diff = ft_v - pt_v
            arrow = "▲" if diff > 0.001 else ("▼" if diff < -0.001 else "─")
            print(f"  {cname:<12}  {ft_v:>14.4f}  {pt_v:>14.4f}  {arrow}{abs(diff):.4f}")
    print(f"{'═'*52}\n")
    return rows


def plot_comparison(rows, res_ft, res_pt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fine-tuned vs Pre-trained  (val set, person+vehicle)",
                 fontsize=13, fontweight="bold")

    # ① 전체 지표 비교
    ax = axes[0]
    metrics = list(rows.keys())
    ft_vals = [rows[m][0] for m in metrics]
    pt_vals = [rows[m][1] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, ft_vals, w, label="fine-tuned",  color="#1976D2", alpha=0.88)
    b2 = ax.bar(x + w/2, pt_vals, w, label="pre-trained", color="#F57C00", alpha=0.88)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_title("Overall Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ② 클래스별 mAP@0.5:0.95
    ax = axes[1]
    if "map_per_class" in res_ft and res_ft["map_per_class"] is not None:
        class_names = list(COMPARE_CLASSES.values())
        ft_per = res_ft["map_per_class"].tolist()
        pt_per = res_pt["map_per_class"].tolist()
        x2 = np.arange(len(class_names))
        b3 = ax.bar(x2 - w/2, ft_per[:len(class_names)], w,
                    label="fine-tuned",  color="#1976D2", alpha=0.88)
        b4 = ax.bar(x2 + w/2, pt_per[:len(class_names)], w,
                    label="pre-trained", color="#F57C00", alpha=0.88)
        for bars in [b3, b4]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x2)
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_title("Per-Class mAP@0.5:0.95")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = SAVE_DIR / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  차트 저장 → {out}")
    if HAS_DISPLAY:
        plt.show()
    plt.close(fig)


def main():
    import cv2
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  val 세트 로드 중...")
    val_data = load_val_data(VAL_TXT, LABELS_DIR)
    print(f"  GT 있는 val 이미지: {len(val_data)}장  (person/vehicle 포함)")

    # ── Fine-tuned 평가 ────────────────────────────────────────────────────
    print(f"\n[1/2] Fine-tuned 모델 평가 중...  ({FINETUNED_W.name})")
    res_ft = run_eval(FINETUNED_W, val_data, coco_remap=False)

    # ── Pre-trained 평가 ───────────────────────────────────────────────────
    print(f"\n[2/2] Pre-trained 모델 평가 중...  ({PRETRAINED_W.name})")
    res_pt = run_eval(PRETRAINED_W, val_data, coco_remap=True)

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    rows = print_comparison(res_ft, res_pt)
    plot_comparison(rows, res_ft, res_pt)

    # JSON 저장
    out = {
        "fine_tuned":  {k: float(v) for k, v in res_ft.items()
                        if isinstance(v, torch.Tensor) and v.numel() == 1},
        "pre_trained": {k: float(v) for k, v in res_pt.items()
                        if isinstance(v, torch.Tensor) and v.numel() == 1},
    }
    (SAVE_DIR / "comparison.json").write_text(json.dumps(out, indent=2))
    print(f"  수치 저장 → {SAVE_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
