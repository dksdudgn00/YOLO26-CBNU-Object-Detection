import os
import cv2
import numpy as np
from pathlib import Path

# ================================================================
# 설정값
# ================================================================
INPUT_ROOT = "/home/a/Desktop/yhan/Sources/custom_yolo_train/datasets/origin_images"   # bag1, bag2, bag3 폴더가 있는 루트
OUTPUT_DIR = "/home/a/Desktop/yhan/Sources/custom_yolo_train/datasets/filtered_images" # 필터링된 결과 저장 폴더
SIMILARITY_THRESHOLD = 0.92      # 유사도 임계값 (높을수록 엄격하게 중복 제거)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_hash(img, size=(16, 16)):
    """
    이미지를 작은 해상도로 줄인 뒤 평균값 기준으로 흑백 해시 생성
    - 빠른 유사도 비교를 위해 사용
    - size: 줄일 해상도 (작을수록 빠르지만 덜 정밀)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    resized = cv2.resize(gray, size)               # 작게 리사이즈
    avg = resized.mean()                           # 평균 픽셀값
    return (resized > avg).flatten()               # 평균보다 밝으면 1, 어두우면 0

def hash_similarity(h1, h2):
    """
    두 해시 벡터의 유사도 계산 (같은 비트 비율)
    - 1.0이면 완전히 동일, 0.0이면 완전히 다름
    """
    return np.sum(h1 == h2) / len(h1)

# ================================================================
# 각 bag 폴더별로 처리
# ================================================================
total_saved = 0
total_skipped = 0

# bag1, bag2, bag3 ... 폴더 순회
bag_folders = sorted([
    d for d in Path(INPUT_ROOT).iterdir() if d.is_dir()
])

for bag_folder in bag_folders:
    print(f"\n📁 처리 중: {bag_folder.name}")
    
    # 해당 폴더의 출력 디렉토리 생성
    out_subdir = Path(OUTPUT_DIR) / bag_folder.name
    os.makedirs(out_subdir, exist_ok=True)
    
    # 이미지 파일 목록 수집
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths = sorted([
        p for p in bag_folder.iterdir()
        if p.suffix.lower() in img_extensions
    ])
    
    saved_hashes = []   # 저장된 이미지들의 해시 목록
    saved = 0
    skipped = 0
    
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        curr_hash = get_image_hash(img)
        
        # 이미 저장된 이미지들과 유사도 비교
        is_duplicate = False
        for prev_hash in saved_hashes:
            sim = hash_similarity(curr_hash, prev_hash)
            if sim >= SIMILARITY_THRESHOLD:
                # 유사도가 임계값 이상이면 중복으로 판단하고 스킵
                is_duplicate = True
                break
        
        if not is_duplicate:
            # 중복 아니면 저장
            out_path = out_subdir / img_path.name
            cv2.imwrite(str(out_path), img)
            saved_hashes.append(curr_hash)
            saved += 1
        else:
            skipped += 1
    
    print(f"  저장: {saved}장 / 스킵: {skipped}장")
    total_saved += saved
    total_skipped += skipped

print(f"\n✅ 완료 — 최종 저장: {total_saved}장 / 제거: {total_skipped}장")