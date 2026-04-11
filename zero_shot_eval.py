"""
zero_shot_eval.py
-----------------
Đánh giá CLIP zero-shot (không fine-tune) trên DAiSEE test set.
Mục đích: Xác định CLIP base model bias về class nào trước khi training.

Kết quả sẽ cho biết:
  - CLIP có natural bias sang class nào không (thường là class 1 "High")
  - WAR/UAR zero-shot = baseline tối thiểu có thể đạt
  - Prompt nào tốt nhất để phân biệt 3 classes

Chạy:
  python3 zero_shot_eval.py \
      --ann /kaggle/input/.../DAiSEE_data/Labels/TestLabels.csv \
      --root /kaggle/input/.../DAiSEE_data \
      --clip-model ViT-B/16
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict, Counter
import cv2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ann', required=True, help='TestLabels.csv')
    p.add_argument('--root', required=True, help='DAiSEE root dir')
    p.add_argument('--clip-model', default='ViT-B/16')
    p.add_argument('--num-frames', type=int, default=4)
    p.add_argument('--merge-3class', type=bool, default=True)
    return p.parse_args()


def load_clip(model_name, device):
    from clip import clip as clip_lib
    model, preprocess = clip_lib.load(model_name, device=device)
    model.eval()
    return model, preprocess


def get_text_embeddings(model, prompts_per_class, device):
    """Encode prompts for each class, average ensemble."""
    import clip as clip_lib
    class_embeddings = []
    with torch.no_grad():
        for prompts in prompts_per_class:
            tokens = clip_lib.tokenize(prompts).to(device)
            emb = model.encode_text(tokens)  # (N_prompts, D)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0)
            emb = emb / emb.norm()
            class_embeddings.append(emb)
    return torch.stack(class_embeddings)  # (num_classes, D)


def preprocess_frame(img_pil, preprocess):
    return preprocess(img_pil).unsqueeze(0)


def load_samples(ann_path, root, merge_3class=True):
    import pandas as pd
    df = pd.read_csv(ann_path)
    df.columns = df.columns.str.strip()
    samples = []
    for _, row in df.iterrows():
        clip_id_ext = row['ClipID']
        if not isinstance(clip_id_ext, str):
            continue
        label = int(row['Engagement'])
        if merge_3class and label == 0:
            label = 1  # Very Low → Low
        if merge_3class:
            label = max(0, label - 1)  # shift: 1→0, 2→1, 3→2
        clip_id = os.path.splitext(clip_id_ext)[0]
        subject_id = clip_id[:6]
        split_dir = 'Test'
        frames_dir = os.path.join(root, 'DataSet', split_dir, subject_id, clip_id, 'frames')
        if os.path.isdir(frames_dir):
            files = sorted([f for f in os.listdir(frames_dir)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            if files:
                samples.append((frames_dir, files, label))
    return samples


def evaluate_zero_shot(model, preprocess, text_emb, samples, num_frames, device):
    all_preds, all_labels = [], []
    logit_scale = model.logit_scale.exp()

    for frames_dir, files, label in tqdm(samples, desc='Zero-shot eval'):
        # Sample N frames uniformly
        indices = np.linspace(0, len(files)-1, num_frames, dtype=int)
        frame_imgs = []
        for idx in indices:
            try:
                img = Image.open(os.path.join(frames_dir, files[idx])).convert('RGB')
                frame_imgs.append(preprocess(img))
            except:
                pass
        if not frame_imgs:
            continue

        frames_tensor = torch.stack(frame_imgs).to(device)  # (T, C, H, W)
        with torch.no_grad():
            vis_emb = model.encode_image(frames_tensor)  # (T, D)
            vis_emb = vis_emb / vis_emb.norm(dim=-1, keepdim=True)
            vis_emb_mean = vis_emb.mean(dim=0, keepdim=True)  # (1, D)
            vis_emb_mean = vis_emb_mean / vis_emb_mean.norm(dim=-1, keepdim=True)
            logits = (logit_scale * vis_emb_mean @ text_emb.T).squeeze(0)  # (C,)
            pred = logits.argmax().item()

        all_preds.append(pred)
        all_labels.append(label)

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, class_names):
    from sklearn.metrics import confusion_matrix, classification_report
    import sklearn.metrics as metrics

    n = len(labels)
    war = (preds == labels).sum() / n * 100

    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-6) * 100
    uar = per_class_recall.mean()

    print(f"\n{'='*55}")
    print(f"  CLIP Zero-Shot Results ({len(class_names)}-class)")
    print(f"{'='*55}")
    print(f"  WAR (overall acc): {war:.2f}%")
    print(f"  UAR (macro recall): {uar:.2f}%")
    print(f"\n  Per-class Recall:")
    for i, (name, r) in enumerate(zip(class_names, per_class_recall)):
        bar = '█' * int(r / 5)
        print(f"    Class {i} ({name:12s}): {r:5.1f}%  {bar}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Prediction Distribution:")
    pred_counts = Counter(preds.tolist())
    for c in range(len(class_names)):
        pct = 100 * pred_counts.get(c, 0) / n
        print(f"    Predict class {c} ({class_names[c]:12s}): {pred_counts.get(c,0):4d} ({pct:.1f}%)")

    # Bias analysis
    majority_pred = max(pred_counts, key=pred_counts.get)
    if pred_counts.get(majority_pred, 0) / n > 0.7:
        print(f"\n  ⚠️  BIAS DETECTED: {pred_counts.get(majority_pred,0)/n*100:.0f}% predictions = class {majority_pred} ({class_names[majority_pred]})")
        print(f"     CLIP zero-shot is biased. Fine-tuning must overcome this bias.")
    else:
        print(f"\n  ✅ No strong zero-shot bias detected.")

    return war, uar


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 3-class prompts (merged)
    class_names = ['Low', 'High', 'Very High']
    prompts = [
        [  # Low (merged Very Low + Low)
            "A student with eyes closed or nearly closed, head drooping, asleep.",
            "A student with open eyes looking sideways, bored, not watching screen.",
            "A student with heavy droopy eyelids, yawning, distracted.",
            "A student looking away from the screen with a blank unfocused stare.",
        ],
        [  # High
            "A student with open eyes looking directly at the screen, calm neutral face.",
            "A student quietly watching online content with forward-looking eyes.",
            "A student maintaining eye contact with the camera, upright posture.",
            "A student with clear open eyes fixed on the screen, relaxed expression.",
        ],
        [  # Very High
            "A student leaning forward with wide-open eyes and raised eyebrows, excited.",
            "A student with unusually wide alert eyes staring intensely at the screen.",
            "A student visibly excited by content, widened eyes and animated expression.",
            "A student with raised eyebrows and bright eyes, deeply engaged.",
        ],
    ]

    print("Loading CLIP...")
    model, preprocess = load_clip(args.clip_model, device)

    print("Encoding text prompts...")
    text_emb = get_text_embeddings(model, prompts, device)

    print("Loading test samples...")
    samples = load_samples(args.ann, args.root, merge_3class=args.merge_3class)
    print(f"Loaded {len(samples)} test clips")

    print("Running zero-shot evaluation...")
    preds, labels = evaluate_zero_shot(model, preprocess, text_emb, samples, args.num_frames, device)

    war, uar = compute_metrics(preds, labels, class_names)

    print(f"\n{'='*55}")
    print(f"  Baseline để beat khi fine-tuning:")
    print(f"    Zero-shot WAR = {war:.2f}% | UAR = {uar:.2f}%")
    print(f"  Target WAR > 60%: cần cải thiện thêm {60-war:.1f}pp")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
