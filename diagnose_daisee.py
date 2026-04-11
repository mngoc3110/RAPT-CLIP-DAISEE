"""
diagnose_daisee.py
------------------
Phân tích nhanh bộ dữ liệu DAiSEE:
  - Phân bố class thực tế (train / test)
  - Tỷ lệ face detection thành công (Haar Cascade)
  - Kiểm tra clip nào bị blank/miss
  - Đề xuất crop strategy

Chạy:
  python3 diagnose_daisee.py \
      --ann-train /kaggle/input/datasets/.../Labels/TrainLabels.csv \
      --ann-test  /kaggle/input/datasets/.../Labels/TestLabels.csv \
      --root-dir  /kaggle/input/datasets/.../DAiSEE_data \
      --sample-check 100
"""

import os
import argparse
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

# CLIP DETECTION
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ann-train', required=True)
    p.add_argument('--ann-test',  required=True)
    p.add_argument('--root-dir',  required=True)
    p.add_argument('--sample-check', type=int, default=100,
                   help='Số clips để check face detection (mặc định 100)')
    return p.parse_args()


def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def class_distribution(df, label_col='Engagement', split_name='Train'):
    counts = Counter(df[label_col].astype(int).tolist())
    total = sum(counts.values())
    print(f"\n{'='*50}")
    print(f"📊 Class Distribution — {split_name}")
    print(f"{'='*50}")
    class_names = {0: 'Very Low', 1: 'Low', 2: 'High', 3: 'Very High'}
    for c in sorted(counts.keys()):
        n = counts[c]
        pct = 100 * n / total
        bar = '█' * int(pct / 2)
        print(f"  Class {c} ({class_names.get(c, '?'):10s}): {n:6d} ({pct:5.1f}%)  {bar}")
    print(f"  Total: {total}")

    # WARNING thresholds
    min_count = min(counts.values())
    max_count = max(counts.values())
    ratio = max_count / (min_count + 1e-6)
    if ratio > 50:
        print(f"\n  ⚠️  CRITICAL: Imbalance ratio {ratio:.0f}:1 — WeightedSampler will MEMORIZE minority class!")
    elif ratio > 10:
        print(f"\n  ⚠️  WARNING: Imbalance ratio {ratio:.0f}:1 — Consider merging or focal loss.")
    else:
        print(f"\n  ✅ Imbalance ratio {ratio:.1f}:1 — Acceptable.")
    return counts


def resolve_clip_path(root_dir, split_dir, clip_id_ext):
    clip_id = os.path.splitext(clip_id_ext)[0]
    subject_id = clip_id[:6]
    clip_dir = os.path.join(root_dir, 'DataSet', split_dir, subject_id, clip_id)
    
    # Try frames/ subdir first
    frames_path = os.path.join(clip_dir, 'frames')
    if os.path.isdir(frames_path):
        files = sorted([f for f in os.listdir(frames_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if files:
            return ('frames', frames_path, files)
    
    # Try video file
    video_path = os.path.join(clip_dir, clip_id_ext)
    if os.path.isfile(video_path):
        return ('video', video_path, None)
    
    # Try any .avi
    try:
        avis = [f for f in os.listdir(clip_dir) if f.endswith('.avi')]
        if avis:
            return ('video', os.path.join(clip_dir, avis[0]), None)
    except:
        pass
    return ('missing', clip_dir, None)


def check_face_detection(root_dir, df, split_dir, n_sample=100):
    """Sample N clips, load 1 frame each, test Haar Cascade."""
    rows = df.sample(min(n_sample, len(df)), random_state=42)
    results = {'found': 0, 'fallback': 0, 'missing': 0, 'total': 0}
    
    print(f"\n{'='*50}")
    print(f"🔍 Face Detection Check ({n_sample} clips, split={split_dir})")
    print(f"{'='*50}")
    
    for _, row in rows.iterrows():
        clip_id_ext = row['ClipID']
        if not isinstance(clip_id_ext, str):
            continue
        
        src = resolve_clip_path(root_dir, split_dir, clip_id_ext)
        source_type, path, files = src
        results['total'] += 1
        
        if source_type == 'missing':
            results['missing'] += 1
            continue
        
        # Load one frame
        img_pil = None
        if source_type == 'frames' and files:
            try:
                img_pil = Image.open(os.path.join(path, files[len(files)//2])).convert('RGB')
            except:
                pass
        elif source_type == 'video':
            cap = cv2.VideoCapture(path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if img_pil is None:
            results['missing'] += 1
            continue
        
        # Run Haar Cascade
        img_cv = np.array(img_pil)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
        
        if len(faces) > 0:
            results['found'] += 1
        else:
            results['fallback'] += 1
    
    valid = results['found'] + results['fallback']
    if valid > 0:
        detection_rate = 100 * results['found'] / valid
        print(f"  ✅ Face detected   : {results['found']:4d} ({detection_rate:.1f}% of valid clips)")
        print(f"  ⚠️  Fallback crop   : {results['fallback']:4d} ({100-detection_rate:.1f}% of valid clips)")
        print(f"  ❌ Missing/corrupt  : {results['missing']:4d}")
        
        if detection_rate < 60:
            print("\n  ‼️  CRITICAL: Haar Cascade detection rate < 60%!")
            print("     → Phần lớn clips dùng fallback center crop (60%).")
            print("     → Dual-stream nhận cùng input → không có information gain.")
            print("     → Đề xuất: Thay bằng MediaPipe Face Detection.")
        elif detection_rate < 80:
            print("\n  ⚠️  WARNING: Detection rate thấp. Cân nhắc MediaPipe hoặc giảm minNeighbors=3.")
        else:
            print("\n  ✅ Detection rate OK.")
    
    return results


def suggest_strategy(train_counts):
    """Đề xuất chiến lược dựa trên phân bố."""
    print(f"\n{'='*50}")
    print("💡 Đề Xuất Chiến Lược")
    print(f"{'='*50}")
    
    class0 = train_counts.get(0, 0)
    class1 = train_counts.get(1, 0)
    class2 = train_counts.get(2, 0)
    class3 = train_counts.get(3, 0)
    
    if class0 < 100:
        print("""
  ✅ KHUYẾN NGHỊ MẠNH: Dùng merge_3class=True (3-class thay vì 4-class)
     Lý do: Class 0 (Very Low) chỉ có {:d} mẫu → quá ít để học được.
     Merge Very Low + Low → "Low" giúp:
       - Tăng mẫu minority class từ {:d} → {:d}
       - Model học pattern "phân tâm" thay vì memorize
       - WAR và UAR đều tăng đáng kể
     """.format(class0, class0, class0 + class1))
    
    total = class0 + class1 + class2 + class3
    class2_pct = 100 * class2 / (total + 1e-6)
    if class2_pct > 60:
        print(f"""
  ⚠️  Class 2 (High) chiếm {class2_pct:.0f}% dataset.
     → KHÔNG dùng WeightedRandomSampler (sẽ oversample class 0 → overfit).
     → Dùng Focal Loss (gamma=2.0-3.0) không cần sampler.
     → Model sẽ đạt WAR cao hơn vì học được phân bố tự nhiên.
     """)
    
    print("""
  📋 Script training tối ưu đề xuất:
     --loss-type focal --focal-gamma 2.0
     --temperature 0.1              (thay 0.07)
     --temporal-layers 2            (thay 1)
     --num-segments 12              (thay 8)
     --scheduler cosine --warmup-epochs 5
     --mixup-alpha 0.2
     [BỎ] --use-weighted-sampler
     """)


def main():
    args = parse_args()
    
    print("\n" + "="*50)
    print("🏥 DAiSEE Dataset Diagnosis")
    print("="*50)
    
    # Load annotations
    train_df = load_df(args.ann_train)
    test_df  = load_df(args.ann_test)
    
    # Class distribution
    train_counts = class_distribution(train_df, split_name='Train')
    test_counts  = class_distribution(test_df,  split_name='Test')
    
    # Face detection check
    check_face_detection(args.root_dir, train_df, 'Train', n_sample=args.sample_check)
    
    # Strategy suggestions
    suggest_strategy(train_counts)
    
    print("\n✅ Diagnosis complete.")


if __name__ == '__main__':
    main()
