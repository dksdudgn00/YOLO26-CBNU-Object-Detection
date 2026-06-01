#!/usr/bin/env python3
"""
Validation script for YOLO26 — run after training is complete.
Evaluates best.pt on the held-out val set and saves full metrics + plots.
"""

import os
import csv
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
WORK_DIR     = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
WEIGHTS      = WORK_DIR / "runs/yolo26_train/weights/best.pt"  # or last.pt
DATA_YAML    = WORK_DIR / "datasets/data.yaml"
VAL_TXT      = WORK_DIR / "datasets/val.txt"
CLASSES      = ["person", "tree", "vehicle"]
IMG_SIZE     = 640
CONF_THRESH  = 0.25   # confidence threshold
IOU_THRESH   = 0.50   # IoU threshold for NMS
SAVE_DIR     = WORK_DIR / "validation_results"
# ─────────────────────────────────────────────────────────────────────────────

HAS_DISPLAY = bool(os.environ.get("DISPLAY", ""))
matplotlib.use("TkAgg" if HAS_DISPLAY else "Agg")


def run_validation():
    import torch
    from ultralytics import YOLO

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*58}")
    print(f"  YOLO26 Validation")
    print(f"  Weights : {WEIGHTS}")
    print(f"  Val set : {VAL_TXT}")
    print(f"  Device  : {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")
    print(f"{'='*58}\n")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}\n"
                                "Training might still be in progress.")

    model = YOLO(str(WEIGHTS))

    # ── Run validation ────────────────────────────────────────────────────────
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        device=device,
        workers=0,
        save_json=True,
        plots=True,                          # saves confusion matrix, PR curve, etc.
        project=str(SAVE_DIR),
        name="run",
        exist_ok=True,
        verbose=True,
    )

    return metrics


def print_metrics(metrics):
    """Pretty-print per-class and overall metrics."""
    box = metrics.box

    print(f"\n{'─'*58}")
    print(f"  {'Class':<15}  {'Prec':>7}  {'Recall':>7}  {'mAP@50':>8}  {'mAP@50:95':>10}")
    print(f"{'─'*58}")

    # Per-class metrics (ultralytics stores them in box.ap_class_index etc.)
    class_metrics = []
    if hasattr(box, 'ap_class_index') and box.ap_class_index is not None:
        for i, cls_idx in enumerate(box.ap_class_index):
            name = CLASSES[cls_idx] if cls_idx < len(CLASSES) else f"class{cls_idx}"
            p  = float(box.p[i])   if hasattr(box, 'p')  else 0.0
            r  = float(box.r[i])   if hasattr(box, 'r')  else 0.0
            ap50   = float(box.ap50[i])   if hasattr(box, 'ap50')   else 0.0
            ap5095 = float(box.ap[i])     if hasattr(box, 'ap')     else 0.0
            print(f"  {name:<15}  {p:>7.4f}  {r:>7.4f}  {ap50:>8.4f}  {ap5095:>10.4f}")
            class_metrics.append({
                "class": name,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mAP50": round(ap50, 4),
                "mAP50_95": round(ap5095, 4),
            })
    print(f"{'─'*58}")

    # Overall
    mp     = float(box.mp)     if hasattr(box, 'mp')     else 0.0
    mr     = float(box.mr)     if hasattr(box, 'mr')     else 0.0
    map50  = float(box.map50)  if hasattr(box, 'map50')  else 0.0
    map    = float(box.map)    if hasattr(box, 'map')    else 0.0
    print(f"  {'all':<15}  {mp:>7.4f}  {mr:>7.4f}  {map50:>8.4f}  {map:>10.4f}")
    print(f"{'─'*58}\n")

    overall = {
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map, 4),
    }
    return class_metrics, overall


def save_metrics_json(class_metrics, overall):
    out = {"overall": overall, "per_class": class_metrics}
    path = SAVE_DIR / "metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  Metrics saved → {path}")


def plot_per_class_map(class_metrics):
    """Bar chart: per-class mAP@50 and mAP@50:95."""
    if not class_metrics:
        return

    names   = [m["class"]    for m in class_metrics]
    map50   = [m["mAP50"]    for m in class_metrics]
    map5095 = [m["mAP50_95"] for m in class_metrics]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 5))
    b1 = ax.bar(x - w/2, map50,   w, label="mAP@0.5",       color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + w/2, map5095, w, label="mAP@0.5:0.95",  color="#FF5722", alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mAP")
    ax.set_title("Per-Class mAP (Validation)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = SAVE_DIR / "per_class_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Per-class mAP chart → {path}")
    if HAS_DISPLAY:
        plt.show()
    plt.close(fig)


def plot_precision_recall(class_metrics, overall):
    """Scatter plot: Precision vs Recall per class."""
    if not class_metrics:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(class_metrics)))

    for m, color in zip(class_metrics, colors):
        ax.scatter(m["recall"], m["precision"], s=120, color=color,
                   zorder=5, label=m["class"])
        ax.annotate(m["class"], (m["recall"], m["precision"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Overall point
    ax.scatter(overall["recall"], overall["precision"], s=180,
               color="black", marker="*", zorder=6, label="overall")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs Recall (Validation)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = SAVE_DIR / "precision_recall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Precision-Recall chart → {path}")
    if HAS_DISPLAY:
        plt.show()
    plt.close(fig)


def plot_training_curve():
    """Re-plot the training mAP curve from results.csv."""
    csv_path = WORK_DIR / "runs/yolo26_train/results.csv"
    if not csv_path.exists():
        return

    epochs, map50_vals, map5095_vals = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                epochs.append(int(float(row.get("epoch", 0))))
                map50_vals.append(float(row.get("metrics/mAP50(B)", 0)))
                map5095_vals.append(float(row.get("metrics/mAP50-95(B)", 0)))
            except ValueError:
                continue

    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, map50_vals,   "b-o", ms=3, lw=1.5, label="mAP@0.5")
    ax.plot(epochs, map5095_vals, "r-o", ms=3, lw=1.5, label="mAP@0.5:0.95")

    best_ep = epochs[int(np.argmax(map50_vals))]
    best_v  = max(map50_vals)
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.6,
               label=f"best epoch ({best_ep})")
    ax.annotate(f"{best_v:.4f}", (best_ep, best_v),
                textcoords="offset points", xytext=(6, -12),
                color="blue", fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Training mAP Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = SAVE_DIR / "training_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Training curve → {path}")
    if HAS_DISPLAY:
        plt.show()
    plt.close(fig)


def main():
    metrics = run_validation()
    class_metrics, overall = print_metrics(metrics)

    save_metrics_json(class_metrics, overall)
    plot_per_class_map(class_metrics)
    plot_precision_recall(class_metrics, overall)
    plot_training_curve()

    print(f"\n  모든 결과 저장 위치: {SAVE_DIR}/")
    print(f"  best weight      : {WEIGHTS}")


if __name__ == "__main__":
    main()
