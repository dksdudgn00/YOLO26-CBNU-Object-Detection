#!/usr/bin/env python3
"""
GT 없이 공정한 성능 비교
─────────────────────────────────────────────────────
Teacher (yolo26m.pt, COCO pretrained, 더 큰 모델)가
no_annotation 이미지에서 고신뢰도(≥TEACHER_CONF) 예측을
'pseudo-GT'로 사용.

두 모델 모두 이 pseudo-GT 기준으로 mAP 측정.
→ 학습 데이터 편향 없음, 둘 다 같은 기준 적용
"""

import os, json, cv2, torch, numpy as np
import matplotlib, matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
TEACHER_W     = Path("/home/a/Downloads/cenDet-main/yolo26m.pt")   # 심판 모델
FINETUNED_W   = WORK_DIR / "runs/yolo26_train/weights/best.pt"
PRETRAINED_W  = WORK_DIR / "yolo26n.pt"
IMAGE_DIR     = WORK_DIR / "datasets/no_annotation/origin_images"
SAVE_DIR      = WORK_DIR / "fair_comparison"
BAG           = "bag1"          # 평가할 bag (bag1 / bag2 / bag3)
N_IMAGES      = 200             # 평가할 이미지 수 (많을수록 정확)
TEACHER_CONF  = 0.60            # teacher pseudo-GT 최소 신뢰도
EVAL_CONF     = 0.25            # 평가 대상 모델 threshold
IOU_NMS       = 0.45

# COCO → 우리 클래스 매핑 (teacher & pre-trained 공통)
COCO_TO_OURS  = {0: 0, 1: 2, 2: 2, 3: 2, 5: 2, 7: 2}
CLASS_NAMES   = {0: "person", 2: "vehicle"}
# ──────────────────────────────────────────────────────────────────────────────

HAS_DISPLAY = bool(os.environ.get("DISPLAY", ""))
matplotlib.use("TkAgg" if HAS_DISPLAY else "Agg")


def collect_images():
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(
        (p for p in (IMAGE_DIR / BAG).iterdir() if p.suffix.lower() in exts),
        key=lambda p: p.stem
    )
    step = max(1, len(imgs) // N_IMAGES)
    return imgs[::step][:N_IMAGES]


def predict(model, img_bgr, device, coco_remap=False):
    coco_classes = list(COCO_TO_OURS.keys()) if coco_remap else None
    res = model(img_bgr, conf=EVAL_CONF, iou=IOU_NMS,
                device=device, verbose=False, classes=coco_classes)[0]
    boxes, labels, scores = [], [], []
    if res.boxes is not None:
        for box in res.boxes:
            cid_raw = int(box.cls[0])
            cid = COCO_TO_OURS.get(cid_raw) if coco_remap else cid_raw
            if cid not in CLASS_NAMES:
                continue
            boxes.append(box.xyxy[0].cpu().tolist())
            labels.append(cid)
            scores.append(float(box.conf[0]))
    return boxes, labels, scores


def teacher_pseudo_gt(model, img_bgr, device):
    """Teacher의 고신뢰도 예측만 pseudo-GT로 사용"""
    coco_classes = list(COCO_TO_OURS.keys())
    res = model(img_bgr, conf=TEACHER_CONF, iou=IOU_NMS,
                device=device, verbose=False, classes=coco_classes)[0]
    boxes, labels = [], []
    if res.boxes is not None:
        for box in res.boxes:
            cid = COCO_TO_OURS.get(int(box.cls[0]))
            if cid not in CLASS_NAMES:
                continue
            boxes.append(box.xyxy[0].cpu().tolist())
            labels.append(cid)
    return boxes, labels


def to_tensor_dict(boxes, labels, scores=None):
    b = torch.tensor(boxes,  dtype=torch.float32) if boxes \
        else torch.zeros((0, 4), dtype=torch.float32)
    l = torch.tensor(labels, dtype=torch.int64)
    d = {"boxes": b, "labels": l}
    if scores is not None:
        d["scores"] = torch.tensor(scores, dtype=torch.float32)
    return d


def run_fair_eval():
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*62}")
    print(f"  공정 비교  —  Teacher pseudo-GT 방식")
    print(f"  Teacher    : {TEACHER_W.name}  (conf≥{TEACHER_CONF})")
    print(f"  Fine-tuned : {FINETUNED_W.name}")
    print(f"  Pre-trained: {PRETRAINED_W.name}")
    print(f"  Images     : {BAG} / {N_IMAGES}장 샘플링")
    print(f"{'='*62}\n")

    images = collect_images()
    print(f"  실제 평가 이미지: {len(images)}장\n")

    print("  모델 로드 중...")
    model_teacher = YOLO(str(TEACHER_W))
    model_ft      = YOLO(str(FINETUNED_W))
    model_pt      = YOLO(str(PRETRAINED_W))

    metric_ft = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric_pt = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    skipped = 0
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Teacher pseudo-GT 생성
        gt_boxes, gt_labels = teacher_pseudo_gt(model_teacher, img, device)
        if not gt_boxes:          # teacher가 아무것도 못 잡으면 스킵
            skipped += 1
            continue
        gt_dict = to_tensor_dict(gt_boxes, gt_labels)

        # Fine-tuned 예측
        b, l, s = predict(model_ft, img, device, coco_remap=False)
        metric_ft.update([to_tensor_dict(b, l, s)], [gt_dict])

        # Pre-trained 예측
        b, l, s = predict(model_pt, img, device, coco_remap=True)
        metric_pt.update([to_tensor_dict(b, l, s)], [gt_dict])

        if idx % 20 == 0 or idx == len(images):
            print(f"  [{idx:3d}/{len(images)}]  pseudo-GT 있는 프레임: {idx-skipped}",
                  end="\r")

    print(f"\n  스킵(teacher 탐지 없음): {skipped}장\n")
    return metric_ft.compute(), metric_pt.compute()


def print_and_plot(res_ft, res_pt):
    keys = [("map_50", "mAP@0.5"), ("map_75", "mAP@0.75"),
            ("map", "mAP@0.5:0.95"), ("mar_100", "Recall@100")]

    print(f"\n{'═'*58}")
    print(f"  ※ 기준: Teacher(yolo26m) 고신뢰도 예측 = pseudo-GT")
    print(f"{'─'*58}")
    print(f"  {'Metric':<16}  {'fine-tuned':>12}  {'pre-trained':>12}  {'차이':>8}")
    print(f"{'─'*58}")

    rows = {}
    for k, label in keys:
        fv = float(res_ft.get(k, 0))
        pv = float(res_pt.get(k, 0))
        diff = fv - pv
        arrow = "▲" if diff > 0.001 else ("▼" if diff < -0.001 else "─")
        print(f"  {label:<16}  {fv:>12.4f}  {pv:>12.4f}  {arrow}{abs(diff):.4f}")
        rows[label] = (fv, pv)

    if "map_per_class" in res_ft and res_ft["map_per_class"] is not None:
        print(f"{'─'*58}")
        print(f"  클래스별 mAP@0.5:0.95")
        ft_per = res_ft["map_per_class"].tolist()
        pt_per = res_pt["map_per_class"].tolist()
        for i, (cid, cname) in enumerate(CLASS_NAMES.items()):
            fv = ft_per[i] if i < len(ft_per) else 0.0
            pv = pt_per[i] if i < len(pt_per) else 0.0
            diff = fv - pv
            arrow = "▲" if diff > 0.001 else ("▼" if diff < -0.001 else "─")
            print(f"  {cname:<16}  {fv:>12.4f}  {pv:>12.4f}  {arrow}{abs(diff):.4f}")
    print(f"{'═'*58}\n")

    # ── 차트 ───────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Fair Comparison  (Teacher: yolo26m conf≥{TEACHER_CONF} / {BAG} {N_IMAGES}장)",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    labels_ = list(rows.keys())
    x = np.arange(len(labels_))
    w = 0.35
    b1 = ax.bar(x-w/2, [rows[l][0] for l in labels_], w,
                label="fine-tuned",  color="#1976D2", alpha=0.88)
    b2 = ax.bar(x+w/2, [rows[l][1] for l in labels_], w,
                label="pre-trained", color="#F57C00", alpha=0.88)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels_, fontsize=9, rotation=15)
    ax.set_ylim(0, 1.1); ax.set_title("Overall"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    if "map_per_class" in res_ft and res_ft["map_per_class"] is not None:
        cnames = list(CLASS_NAMES.values())
        ft_per = res_ft["map_per_class"].tolist()
        pt_per = res_pt["map_per_class"].tolist()
        x2 = np.arange(len(cnames))
        b3 = ax.bar(x2-w/2, ft_per[:len(cnames)], w, label="fine-tuned",  color="#1976D2", alpha=0.88)
        b4 = ax.bar(x2+w/2, pt_per[:len(cnames)], w, label="pre-trained", color="#F57C00", alpha=0.88)
        for bars in [b3, b4]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h+0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x2); ax.set_xticklabels(cnames, fontsize=11)
        ax.set_ylim(0, 1.1); ax.set_title("Per-Class mAP@0.5:0.95")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = SAVE_DIR / "fair_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  차트 저장 → {out}")
    if HAS_DISPLAY:
        plt.show()
    plt.close(fig)

    # JSON
    (SAVE_DIR / "fair_comparison.json").write_text(json.dumps({
        "fine_tuned":  {k: float(v) for k,v in res_ft.items()
                        if isinstance(v, torch.Tensor) and v.numel()==1},
        "pre_trained": {k: float(v) for k,v in res_pt.items()
                        if isinstance(v, torch.Tensor) and v.numel()==1},
        "config": {"teacher": TEACHER_W.name, "teacher_conf": TEACHER_CONF,
                   "bag": BAG, "n_images": N_IMAGES}
    }, indent=2))


def main():
    res_ft, res_pt = run_fair_eval()
    print_and_plot(res_ft, res_pt)
    print(f"  결과 저장 → {SAVE_DIR}/")


if __name__ == "__main__":
    main()
