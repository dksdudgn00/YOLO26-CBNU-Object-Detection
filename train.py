#!/usr/bin/env python3
import os
import random
import shutil
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR   = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train/datasets/train")
MODEL_WEIGHTS = Path("/home/a/Downloads/cenDet-main/yolo26m.pt")
WORK_DIR      = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
CLASSES       = ["person", "tree", "vehicle"]
EPOCHS        = 100
BATCH_SIZE    = 8
IMG_SIZE      = 640
VAL_RATIO     = 0.2
SEED          = 42
# ─────────────────────────────────────────────────────────────────────────────

HAS_DISPLAY = bool(os.environ.get("DISPLAY", ""))
matplotlib.use("TkAgg" if HAS_DISPLAY else "Agg")


# ── Label preprocessing: polygon → bbox ──────────────────────────────────────

def _polygon_to_bbox(fields: list[str]) -> str:
    """[class x1 y1 x2 y2 ...] → YOLO detect [class cx cy w h]"""
    cls = fields[0]
    coords = list(map(float, fields[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prepare_detect_dataset() -> Path:
    """
    Build WORK_DIR/datasets/detect/ with:
      images/ → symlink to original images
      labels/ → converted bbox labels
    Returns the detect/ root Path.
    """
    detect_dir = WORK_DIR / "datasets" / "detect"
    img_link   = detect_dir / "images"
    lbl_dir    = detect_dir / "labels"

    # Images: symlink so ultralytics can find them at detect/images/
    img_link.parent.mkdir(parents=True, exist_ok=True)
    if img_link.exists() or img_link.is_symlink():
        img_link.unlink()
    img_link.symlink_to(DATASET_DIR / "images")

    # Labels: convert polygon → bbox, write to detect/labels/
    if lbl_dir.exists():
        shutil.rmtree(lbl_dir)
    lbl_dir.mkdir(parents=True)

    kept = converted = skipped = 0
    for src in (DATASET_DIR / "labels").glob("*.txt"):
        out_lines = []
        for line in src.read_text().splitlines():
            fields = line.split()
            n = len(fields)
            if n == 0:
                continue
            if n == 5:
                out_lines.append(line)
                kept += 1
            elif n >= 7 and (n - 1) % 2 == 0:
                out_lines.append(_polygon_to_bbox(fields))
                converted += 1
            else:
                skipped += 1
        (lbl_dir / src.name).write_text("\n".join(out_lines) + "\n")

    print(f"  Labels — kept bbox: {kept}  converted polygon→bbox: {converted}  skipped: {skipped}")
    return detect_dir


# ── Dataset split ─────────────────────────────────────────────────────────────

def create_split(detect_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_dir = detect_dir / "images"
    lbl_dir = detect_dir / "labels"

    label_stems = {p.stem for p in lbl_dir.glob("*.txt")}
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in exts and p.stem in label_stems)

    random.seed(SEED)
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_RATIO))
    val_imgs, train_imgs = images[:n_val], images[n_val:]

    split_dir = WORK_DIR / "datasets"
    (split_dir / "train.txt").write_text("\n".join(str(p) for p in train_imgs) + "\n")
    (split_dir / "val.txt").write_text("\n".join(str(p) for p in val_imgs) + "\n")

    print(f"  Split  — train: {len(train_imgs)}  val: {len(val_imgs)}")
    return split_dir / "train.txt", split_dir / "val.txt"


def write_data_yaml(train_txt: Path, val_txt: Path) -> Path:
    path = WORK_DIR / "datasets" / "data.yaml"
    path.write_text(
        f"train: {train_txt}\n"
        f"val:   {val_txt}\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    return path


# ── Real-time mAP plot ────────────────────────────────────────────────────────

_history: dict = defaultdict(list)
_fig = _ax_map = _ax_loss = None


def _init_plot():
    global _fig, _ax_map, _ax_loss
    if HAS_DISPLAY:
        plt.ion()
    _fig, (_ax_map, _ax_loss) = plt.subplots(1, 2, figsize=(13, 5))
    _fig.suptitle("YOLO26 Training Progress", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2.5)
    if HAS_DISPLAY:
        plt.pause(0.001)


def _update_plot(epoch, map50, map5095, losses: dict):
    _history["epoch"].append(epoch)
    _history["mAP50"].append(map50)
    _history["mAP50_95"].append(map5095)
    for k, v in losses.items():
        _history[k].append(v)

    ep = _history["epoch"]

    ax = _ax_map
    ax.clear()
    ax.plot(ep, _history["mAP50"],    "b-o", ms=4, lw=1.5, label="mAP@0.5")
    ax.plot(ep, _history["mAP50_95"], "r-o", ms=4, lw=1.5, label="mAP@0.5:0.95")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = _ax_loss
    ax.clear()
    palette = ["#2ca02c", "#9467bd", "#17becf", "#d62728", "#ff7f0e"]
    loss_keys = [k for k in _history if k not in ("epoch", "mAP50", "mAP50_95")]
    for k, color in zip(loss_keys, palette):
        vals = _history[k]
        ax.plot(ep[: len(vals)], vals, "-o", color=color, ms=4, lw=1.5, label=k)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout(pad=2.5)
    _fig.savefig(WORK_DIR / "training_progress.png", dpi=100, bbox_inches="tight")
    if HAS_DISPLAY:
        _fig.canvas.draw()
        plt.pause(0.001)


# ── Ultralytics callbacks ─────────────────────────────────────────────────────

def _on_fit_epoch_end(trainer):
    m = trainer.metrics

    map50   = m.get("metrics/mAP50(B)",    0.0)
    map5095 = m.get("metrics/mAP50-95(B)", 0.0)

    losses: dict = {}
    if hasattr(trainer, "loss_names") and hasattr(trainer, "tloss"):
        try:
            tloss = trainer.tloss
            if hasattr(tloss, "__iter__"):
                for name, val in zip(trainer.loss_names, tloss):
                    losses[name] = round(float(val), 5)
            else:
                losses["total"] = round(float(tloss), 5)
        except Exception:
            pass
    if not losses:
        losses = {k.replace("train/", ""): round(v, 5)
                  for k, v in m.items() if k.startswith("train/")}

    ep = trainer.epoch + 1
    _update_plot(ep, map50, map5095, losses)

    loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
    print(f"\n  [Epoch {ep:3d}/{trainer.epochs}]  "
          f"mAP@0.5={map50:.4f}  mAP@0.5:0.95={map5095:.4f}  {loss_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*58}")
    print(f"  YOLO26 Detection Training")
    print(f"  Model   : {MODEL_WEIGHTS}")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Device  : {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")
    print(f"{'='*58}\n")

    print("[1/3] Converting polygon labels to bbox format...")
    detect_dir = prepare_detect_dataset()

    print("[2/3] Creating train/val split...")
    train_txt, val_txt = create_split(detect_dir)
    data_yaml = write_data_yaml(train_txt, val_txt)

    print("[3/3] Starting training...\n")
    _init_plot()

    torch.cuda.empty_cache()

    model = YOLO(str(MODEL_WEIGHTS))
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    model.train(
        data=str(data_yaml),
        task="detect",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=0,
        project=str(WORK_DIR / "runs"),
        name="yolo26_train",
        exist_ok=True,
        verbose=True,
    )

    _fig.savefig(WORK_DIR / "training_final.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {WORK_DIR / 'training_final.png'}")

    if HAS_DISPLAY:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
