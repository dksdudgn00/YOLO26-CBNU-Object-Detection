#!/usr/bin/env python3
"""
GT 라벨 vs Pre-trained 예측 시각적 진단
- val 이미지 샘플 10장에 GT(초록)와 pre-trained 예측(파랑)을 함께 그려 저장
- GT 라벨 품질이 이상한지, 모델 예측이 이상한지 눈으로 확인
"""

import cv2
import random
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

WORK_DIR    = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
PRETRAINED  = WORK_DIR / "yolo26n.pt"
FINETUNED   = WORK_DIR / "runs/yolo26_train/weights/best.pt"
VAL_TXT     = WORK_DIR / "datasets/val.txt"
LABELS_DIR  = WORK_DIR / "datasets/detect/labels"
SAVE_DIR    = WORK_DIR / "diagnosis"
N_SAMPLES   = 15
CONF        = 0.25
COCO_TO_OURS = {0:0, 1:2, 2:2, 3:2, 5:2, 7:2}
CLASS_NAMES  = {0:"person", 1:"tree", 2:"vehicle"}

def xywh_to_xyxy(cx,cy,w,h,W,H):
    return int((cx-w/2)*W), int((cy-h/2)*H), int((cx+w/2)*W), int((cy+h/2)*H)

def draw_label(img, x1,y1,x2,y2, text, color, filled=True):
    cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    if filled:
        cv2.rectangle(img,(x1,y1-th-5),(x1+tw+3,y1),color,-1)
        cv2.putText(img,text,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    else:
        cv2.putText(img,text,(x1+2,y1+th+2),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model_pt = YOLO(str(PRETRAINED))
    model_ft = YOLO(str(FINETUNED))

    # val 이미지 목록
    img_paths = [Path(p.strip()) for p in VAL_TXT.read_text().splitlines() if p.strip()]
    # GT 있는 것만
    valid = []
    for p in img_paths:
        lbl = LABELS_DIR / (p.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            valid.append(p)

    samples = random.sample(valid, min(N_SAMPLES, len(valid)))
    print(f"  진단할 이미지: {len(samples)}장\n")

    for img_path in samples:
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue
        H, W = img_orig.shape[:2]

        # GT 라벨 파싱
        gt_boxes = []
        for line in (LABELS_DIR / (img_path.stem+".txt")).read_text().splitlines():
            f = line.split()
            if len(f) != 5: continue
            cls,cx,cy,bw,bh = int(f[0]),float(f[1]),float(f[2]),float(f[3]),float(f[4])
            gt_boxes.append((cls,cx,cy,bw,bh))

        # pre-trained 추론
        res_pt = model_pt(img_orig, conf=CONF, iou=0.45, device=device,
                          verbose=False, classes=list(COCO_TO_OURS.keys()))[0]
        # fine-tuned 추론
        res_ft = model_ft(img_orig, conf=CONF, iou=0.45, device=device,
                          verbose=False)[0]

        # ── 3분할 캔버스: GT | Pre-trained | Fine-tuned ──────────────────
        panels = []
        for title, color_bg in [
            ("GT (ground truth)",          (20,60,20)),
            ("Pre-trained yolo26n",         (20,20,60)),
            ("Fine-tuned best.pt",          (60,20,20)),
        ]:
            panel = img_orig.copy()
            # 제목 바
            cv2.rectangle(panel,(0,0),(W,32),color_bg,-1)
            cv2.putText(panel,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,(255,255,255),1,cv2.LINE_AA)
            panels.append(panel)

        # GT 그리기 (초록)
        for (cls,cx,cy,bw,bh) in gt_boxes:
            x1,y1,x2,y2 = xywh_to_xyxy(cx,cy,bw,bh,W,H)
            name = CLASS_NAMES.get(cls,str(cls))
            draw_label(panels[0],x1,y1,x2,y2,name,(0,200,0))

        # Pre-trained 그리기 (파랑)
        if res_pt.boxes is not None:
            for box in res_pt.boxes:
                cid = COCO_TO_OURS.get(int(box.cls[0]))
                if cid is None: continue
                x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().numpy())
                cf = float(box.conf[0])
                name = CLASS_NAMES.get(cid,str(cid))
                draw_label(panels[1],x1,y1,x2,y2,f"{name} {cf:.2f}",(255,100,0))

        # Fine-tuned 그리기 (주황)
        if res_ft.boxes is not None:
            for box in res_ft.boxes:
                cid = int(box.cls[0])
                x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().numpy())
                cf = float(box.conf[0])
                name = CLASS_NAMES.get(cid,str(cid))
                draw_label(panels[2],x1,y1,x2,y2,f"{name} {cf:.2f}",(0,140,255))

        # 3분할 이어붙이기
        # 각 패널을 640x360으로 리사이즈
        resized = [cv2.resize(p,(640,360)) for p in panels]
        canvas = np.hstack(resized)

        # GT 통계
        n_gt = len(gt_boxes)
        n_pt = len(res_pt.boxes) if res_pt.boxes is not None else 0
        n_ft = len(res_ft.boxes) if res_ft.boxes is not None else 0

        # 하단 정보 바
        bar = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
        bar[:] = (30,30,30)
        info = f"GT:{n_gt}개   Pre-trained:{n_pt}개   Fine-tuned:{n_ft}개    [{img_path.name}]"
        cv2.putText(bar,info,(8,19),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1,cv2.LINE_AA)
        canvas = np.vstack([canvas, bar])

        out_path = SAVE_DIR / f"diag_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  저장: {out_path.name}  |  GT:{n_gt}  PT:{n_pt}  FT:{n_ft}")

    print(f"\n  결과 폴더: {SAVE_DIR}/")
    print("  GT(초록) vs Pre-trained(파랑) vs Fine-tuned(주황) 비교")
    print("  → GT 박스가 이상하면 라벨 품질 문제")
    print("  → PT 예측이 아예 없으면 도메인/threshold 문제")

if __name__ == "__main__":
    main()
