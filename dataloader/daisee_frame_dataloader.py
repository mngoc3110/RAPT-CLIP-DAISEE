"""
DAiSEE Frame-Level Dataloader — treats each frame as an independent sample.
Inspired by CAER-S: single-image classification instead of video-level temporal modeling.

Benefits:
- 10x+ more training samples (each video → multiple frames)
- CLIP works best on single images
- Eliminates temporal complexity
"""
import os
import cv2
import pandas as pd
import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

# Load HaarCascade
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# CLIP normalization
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class DAiSEEFrameDataset(data.Dataset):
    """Frame-level DAiSEE dataset — each frame is an independent sample.
    
    For each video clip, extracts `frames_per_clip` evenly-spaced frames.
    Each frame becomes a separate (face_img, body_img, label) sample.
    
    With ~1756 training clips × 10 frames = ~17,560 training samples.
    """
    
    def __init__(self, root_dir, annotation_file, mode='train',
                 image_size=224, frames_per_clip=10,
                 max_samples_per_class=0, extra_annotation_files=None,
                 merge_3class=False, face_only_mode=False):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.mode = mode
        self.image_size = image_size
        self.frames_per_clip = frames_per_clip
        self.label_col = 'Engagement'
        self.max_samples_per_class = max_samples_per_class
        self.extra_annotation_files = extra_annotation_files or []
        self.merge_3class = merge_3class
        self.face_only_mode = face_only_mode
        
        self.samples = self._make_dataset()
        
        # Transforms (single image, like CAER-S)
        if mode == 'train':
            self.transform_face = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
            self.transform_body = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform_face = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
            self.transform_body = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
    
    def _make_dataset(self):
        """Build frame-level samples: (frame_source, frame_idx, label)
        
        frame_source: ('frames', frames_dir, filename) or ('video', video_path, frame_number) 
        """
        all_clips = []
        files_to_load = [self.annotation_file] + self.extra_annotation_files
        
        for ann_file in files_to_load:
            if not os.path.exists(ann_file):
                print(f"Warning: {ann_file} not found. Skipping.")
                continue
            
            print(f"Loading annotations from: {ann_file}")
            
            # Determine split directory
            if 'Train' in ann_file:
                split_dir = 'Train'
            elif 'Validation' in ann_file:
                split_dir = 'Validation'
            elif 'Test' in ann_file:
                split_dir = 'Test'
            else:
                split_dir = 'Train' if self.mode == 'train' else ('Validation' if self.mode == 'val' else 'Test')
            
            try:
                df = pd.read_csv(ann_file)
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(ann_file)}"):
                    clip_id_ext = row['ClipID']
                    if not isinstance(clip_id_ext, str):
                        continue
                    clip_id = os.path.splitext(clip_id_ext)[0]
                    subject_id = clip_id[:6]
                    label = int(row[self.label_col])
                    
                    if self.merge_3class:
                        label_map = {0: 0, 1: 0, 2: 1, 3: 2}
                        label = label_map[label]
                    
                    clip_dir = os.path.join(self.root_dir, 'DataSet', split_dir, subject_id, clip_id)
                    all_clips.append((clip_dir, label, clip_id_ext))
            except Exception as e:
                print(f"Error reading CSV {ann_file}: {e}")
        
        # Explode clips into individual frames
        print(f"Exploding {len(all_clips)} clips into frame-level samples ({self.frames_per_clip} frames/clip)...")
        all_samples = []
        skipped = 0
        
        for clip_dir, label, clip_id_ext in tqdm(all_clips, desc="Extracting frames"):
            frames_path = os.path.join(clip_dir, 'frames')
            video_path = os.path.join(clip_dir, clip_id_ext)
            
            if os.path.isdir(frames_path):
                # Pre-extracted frames
                files = sorted([f for f in os.listdir(frames_path) 
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                if len(files) == 0:
                    skipped += 1
                    continue
                # Sample evenly spaced frames
                indices = np.linspace(0, len(files) - 1, self.frames_per_clip, dtype=int)
                for idx in indices:
                    all_samples.append(('frames', os.path.join(frames_path, files[idx]), label))
                    
            elif os.path.isfile(video_path):
                # Video file — store (video_path, frame_number, label)
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if total > 0:
                        indices = np.linspace(0, total - 1, self.frames_per_clip, dtype=int)
                        for idx in indices:
                            all_samples.append(('video', video_path, int(idx), label))
                    else:
                        skipped += 1
                else:
                    cap.release()
                    skipped += 1
            else:
                # Try finding any .avi in the clip directory
                try:
                    avis = [f for f in os.listdir(clip_dir) if f.endswith('.avi')]
                    if avis:
                        vp = os.path.join(clip_dir, avis[0])
                        cap = cv2.VideoCapture(vp)
                        if cap.isOpened():
                            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                            if total > 0:
                                indices = np.linspace(0, total - 1, self.frames_per_clip, dtype=int)
                                for idx in indices:
                                    all_samples.append(('video', vp, int(idx), label))
                            else:
                                skipped += 1
                        else:
                            cap.release()
                            skipped += 1
                    else:
                        skipped += 1
                except:
                    skipped += 1
        
        if skipped > 0:
            print(f"  Skipped {skipped} clips (no frames/video found)")
        
        # Normalize sample format: all should be (source_type, path, frame_idx_or_none, label)
        samples = []
        for s in all_samples:
            if s[0] == 'frames':
                # ('frames', filepath, label)
                samples.append(('frames', s[1], -1, s[2]))
            else:
                # ('video', video_path, frame_idx, label)
                samples.append(s)
        
        # Only UNDERSAMPLE majority classes (training only)
        if self.mode == 'train' and self.max_samples_per_class > 0:
            from collections import defaultdict
            per_class = defaultdict(list)
            for s in samples:
                per_class[s[3]].append(s)
            
            target = self.max_samples_per_class
            
            # Print BEFORE balancing
            print(f"\n{'='*60}")
            print(f"  CLASS DISTRIBUTION BEFORE BALANCING:")
            for cls_idx in sorted(per_class.keys()):
                count = len(per_class[cls_idx])
                print(f"    Class {cls_idx}: {count:>6} frames")
            print(f"    Total:  {sum(len(v) for v in per_class.values()):>6} frames")
            print(f"{'='*60}")
            
            samples = []
            for cls_idx in sorted(per_class.keys()):
                cls_samples = per_class[cls_idx]
                if len(cls_samples) > target:
                    # Undersample: cap to target
                    random.shuffle(cls_samples)
                    cls_samples = cls_samples[:target]
                    print(f"  Undersampled class {cls_idx}: {len(per_class[cls_idx])} → {target}")
                # Keep minority classes AS-IS (no oversampling)
                samples.extend(cls_samples)
            random.shuffle(samples)
            
            # Print AFTER balancing
            print(f"\n  CLASS DISTRIBUTION AFTER BALANCING (cap={target}):")
            for cls_idx in sorted(per_class.keys()):
                count = sum(1 for s in samples if s[3] == cls_idx)
                print(f"    Class {cls_idx}: {count:>6} frames")
            print(f"    Total:  {len(samples):>6} frames")
            print(f"{'='*60}\n")
        
        # Print stats
        from collections import Counter
        label_counts = Counter(s[3] for s in samples)
        print(f"DAiSEE Frame ({self.mode}): {len(samples)} total frame-samples, distribution: {dict(sorted(label_counts.items()))}")
        return samples
    
    def _detect_and_crop_face(self, img_pil, fallback_ratio=0.6):
        """Face detection with Haar cascade, fallback to center crop."""
        w, h = img_pil.size
        img_cv = np.array(img_pil)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, fw, fh = largest_face
            margin_factor = 0.2 if fallback_ratio <= 0.6 else 0.4
            margin_w, margin_h = int(fw * margin_factor), int(fh * margin_factor)
            left = max(0, x - margin_w)
            top = max(0, y - margin_h)
            right = min(w, x + fw + margin_w)
            bottom = min(h, y + fh + margin_h)
            return img_pil.crop((left, top, right, bottom))
        
        # Fallback: center crop
        crop_w, crop_h = int(w * fallback_ratio), int(h * fallback_ratio)
        if self.mode == 'train':
            max_jx = int(w * 0.15)
            max_jy = int(h * 0.15)
            jx = random.randint(-max_jx, max_jx)
            jy = random.randint(-max_jy, max_jy)
            scale = random.uniform(0.9, 1.1)
            crop_w = min(int(crop_w * scale), w)
            crop_h = min(int(crop_h * scale), h)
            left = max(0, min((w - crop_w) // 2 + jx, w - crop_w))
            top = max(0, min((h - crop_h) // 2 + jy, h - crop_h))
        else:
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
        return img_pil.crop((left, top, left + crop_w, top + crop_h))
    
    def _load_frame(self, sample):
        """Load a single frame from file or video."""
        source_type, path, frame_idx, label = sample
        
        try:
            if source_type == 'frames':
                img = Image.open(path).convert('RGB')
            else:
                # Video — seek to specific frame
                cap = cv2.VideoCapture(path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    img = Image.new('RGB', (self.image_size, self.image_size))
        except Exception as e:
            img = Image.new('RGB', (self.image_size, self.image_size))
        
        return img
    
    def __getitem__(self, index):
        sample = self.samples[index]
        label = sample[3]
        
        img = self._load_frame(sample)
        
        # Face crop (tight, 0.5)
        img_face = self._detect_and_crop_face(img, 0.5)
        
        # Body crop (wider, 0.75) or same as face
        if self.face_only_mode:
            img_body = self._detect_and_crop_face(img, 0.75)
        else:
            img_body = img_face
        
        # Apply transforms → (C, H, W)
        t_face = self.transform_face(img_face)
        t_body = self.transform_body(img_body)
        
        # Add temporal dim T=1 → (1, C, H, W) to match model input format
        t_face = t_face.unsqueeze(0)
        t_body = t_body.unsqueeze(0)
        
        return t_face, t_body, label
    
    def __len__(self):
        return len(self.samples)
