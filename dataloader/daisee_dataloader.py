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
from dataloader.video_transform import (
    GroupResize, Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip,
    GroupGaussianBlur, GroupRandomGrayscale, GroupRandomErasing
)

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ============================================================
# OpenCV Haar Cascade for lightweight face detection
# ============================================================
_FACE_CASCADE = None

def _get_face_cascade():
    """Lazy-load the Haar Cascade face detector."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def detect_face_region(frame_bgr, fallback_ratio=0.5):
    """Detect face bounding box using Haar Cascade.
    
    Returns (left, top, width, height) of the best face region,
    or a center crop fallback if no face is detected.
    
    Args:
        frame_bgr: BGR numpy array (H, W, 3)
        fallback_ratio: crop ratio for center crop fallback
    Returns:
        (left, top, crop_w, crop_h) tuple
    """
    h, w = frame_bgr.shape[:2]
    cascade = _get_face_cascade()
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Pick the largest face
        areas = [fw * fh for (_, _, fw, fh) in faces]
        best_idx = np.argmax(areas)
        fx, fy, fw, fh = faces[best_idx]
        
        # Expand the face region by 40% on each side for context
        expand = 0.4
        cx, cy = fx + fw // 2, fy + fh // 2
        new_w = int(fw * (1 + 2 * expand))
        new_h = int(fh * (1 + 2 * expand))
        
        left = max(0, cx - new_w // 2)
        top = max(0, cy - new_h // 2)
        crop_w = min(new_w, w - left)
        crop_h = min(new_h, h - top)
        
        return (left, top, crop_w, crop_h)
    
    # Fallback: center crop
    crop_w = int(w * fallback_ratio)
    crop_h = int(h * fallback_ratio)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return (left, top, crop_w, crop_h)


class DAiSEEDataset(data.Dataset):
    def __init__(self, root_dir, annotation_file, mode='train', num_segments=16, 
                 duration=1, image_size=224, max_samples_per_class=0, 
                 num_engagement_levels=3,
                 use_face_detection=False, temporal_dropout=0.0,
                 augment_strength='mild', face_only_mode=False):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.mode = mode
        self.num_segments = num_segments
        self.duration = duration
        self.image_size = image_size
        self.label_col = 'Engagement' 
        self.frame_cache = {}
        self.max_samples_per_class = max_samples_per_class  # 0 = no cap
        self.num_engagement_levels = num_engagement_levels  # 3 = merged, 4 = original
        self.face_only_mode = face_only_mode  # True = body stream uses wider face crop
        
        # === SOTA enhancements ===
        self.use_face_detection = use_face_detection
        self.temporal_dropout = temporal_dropout  # e.g., 0.15 = drop 15% frames
        self.augment_strength = augment_strength  # 'mild' or 'strong'
        
        # Multi-scale crop ratios (SOTA technique from EfficientNet+TCN papers)
        self.crop_ratios = [0.4, 0.5, 0.6] if mode == 'train' else [0.5]
        # Face-only mode uses wider crop for body stream (multi-scale face)
        self.body_crop_ratios = [0.65, 0.75, 0.85] if (mode == 'train' and face_only_mode) else [0.75] if face_only_mode else None
        
        self.samples = self._make_dataset()
        
        if mode == 'train':
            if augment_strength == 'strong':
                # SOTA-level augmentation
                self._color_jitter = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
                )
                self.group_transform = transforms.Compose([
                    GroupGaussianBlur(p=0.1, kernel_range=(3, 7)),
                    GroupRandomGrayscale(p=0.05),
                    GroupResize(image_size),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                    GroupRandomErasing(p=0.15, scale=(0.02, 0.15)),
                ])
            else:
                # Original mild augmentation
                self._color_jitter = transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.1
                )
                self.group_transform = transforms.Compose([
                    GroupResize(image_size),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                ])
        else:
            self.group_transform = transforms.Compose([
                GroupResize(image_size),
                Stack(),
                ToTorchFormatTensor(),
            ])

    def _make_dataset(self):
        samples = []
        if not os.path.exists(self.annotation_file):
            print(f"Error: Annotation file {self.annotation_file} not found.")
            return samples
            
        print(f"Loading annotations from: {self.annotation_file}")
        try:
            df = pd.read_csv(self.annotation_file)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return samples

        if 'Train' in self.annotation_file or self.mode == 'train':
            split_dir = 'Train'
        elif 'Validation' in self.annotation_file or self.mode == 'val':
            split_dir = 'Validation'
        elif 'Test' in self.annotation_file or self.mode == 'test':
            split_dir = 'Test'
        else:
            split_dir = 'Train'

        print(f"Processing entries for {self.mode}...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {self.mode}"):
            clip_id_ext = row['ClipID']
            if not isinstance(clip_id_ext, str):
                continue
            clip_id = os.path.splitext(clip_id_ext)[0]
            subject_id = clip_id[:6]
            label = int(row[self.label_col])
            if self.num_engagement_levels == 3:
                # Merge to 3 classes: VeryLow(0)+Low(1) → 0, High(2) → 1, VeryHigh(3) → 2
                label_map = {0: 0, 1: 0, 2: 1, 3: 2}
                label = label_map[label]
            elif self.num_engagement_levels == 2:
                # Binary: VeryLow(0)+Low(1) → 0 (Not Engaged), High(2)+VeryHigh(3) → 1 (Engaged)
                label_map = {0: 0, 1: 0, 2: 1, 3: 1}
                label = label_map[label]
            # else (4): keep original 4 levels (0=VeryLow, 1=Low, 2=High, 3=VeryHigh)
            clip_dir = os.path.join(self.root_dir, 'DataSet', split_dir, subject_id, clip_id)
            samples.append((clip_dir, label, clip_id_ext))

        # Undersample majority classes (training only)
        if self.mode == 'train' and self.max_samples_per_class > 0:
            from collections import defaultdict
            per_class = defaultdict(list)
            for s in samples:
                per_class[s[1]].append(s)
            samples = []
            for cls_idx in sorted(per_class.keys()):
                cls_samples = per_class[cls_idx]
                if len(cls_samples) > self.max_samples_per_class:
                    random.shuffle(cls_samples)
                    cls_samples = cls_samples[:self.max_samples_per_class]
                samples.extend(cls_samples)
            random.shuffle(samples)
            class_counts = {i: sum(1 for s in samples if s[1]==i) for i in sorted(per_class.keys())}
            print(f"DAiSEE ({self.mode}): After undersampling → {class_counts}")

        print(f"DAiSEE ({self.mode}): Loaded {len(samples)} samples.")
        
        # Print SOTA config summary
        if self.mode == 'train':
            print(f"  SOTA Config: face_det={self.use_face_detection}, "
                  f"temporal_dropout={self.temporal_dropout}, "
                  f"augment={self.augment_strength}, "
                  f"crop_ratios={self.crop_ratios}")
        return samples

    def _get_indices(self, num_frames):
        if self.mode == 'train':
            average_duration = (num_frames - self.duration + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)
            elif num_frames > self.num_segments:
                offsets = np.sort(np.random.randint(num_frames - self.duration + 1, size=self.num_segments))
            else:
                offsets = np.pad(np.array(list(range(num_frames))), (0, self.num_segments - num_frames), "edge")
        else:
            if num_frames > self.num_segments + self.duration - 1:
                tick = (num_frames - self.duration + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.pad(np.array(list(range(num_frames))), (0, self.num_segments - num_frames), "edge")
        return offsets

    def _get_crop_params(self, w, h, crop_ratio):
        """Generate crop parameters. During training, adds random jitter.
        Returns (left, top, crop_w, crop_h) — CONSISTENT for all frames in a video."""
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        
        if self.mode == 'train':
            # Random jitter: ±10% offset from center (reduced from 15%)
            max_jitter_x = int(w * 0.10)
            max_jitter_y = int(h * 0.10)
            jitter_x = random.randint(-max_jitter_x, max_jitter_x)
            jitter_y = random.randint(-max_jitter_y, max_jitter_y)
            # Random scale: 95-105% of crop_ratio (tighter than before)
            scale = random.uniform(0.95, 1.05)
            crop_w = min(int(crop_w * scale), w)
            crop_h = min(int(crop_h * scale), h)
            left = max(0, min((w - crop_w) // 2 + jitter_x, w - crop_w))
            top = max(0, min((h - crop_h) // 2 + jitter_y, h - crop_h))
        else:
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
        
        return left, top, crop_w, crop_h

    def _apply_crop(self, img_pil, crop_params):
        """Apply pre-computed crop parameters to an image."""
        left, top, crop_w, crop_h = crop_params
        return img_pil.crop((left, top, left + crop_w, top + crop_h))

    def _apply_consistent_color_jitter(self, images):
        """Apply the SAME color jitter transform to ALL frames in a video.
        This preserves temporal coherence while still providing augmentation."""
        if not images:
            return images
        # Get the transform parameters once
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            transforms.ColorJitter.get_params(
                self._color_jitter.brightness,
                self._color_jitter.contrast,
                self._color_jitter.saturation,
                self._color_jitter.hue
            )
        # Apply the same transform to all frames
        result = []
        for img in images:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = transforms.functional.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = transforms.functional.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = transforms.functional.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = transforms.functional.adjust_hue(img, hue_factor)
            result.append(img)
        return result

    def _detect_face_from_video(self, video_path):
        """Detect face region from the first valid frame of the video.
        Returns crop params or None if detection fails."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Sample a few frames and try detection
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = [0, total // 4, total // 2]
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                result = detect_face_region(frame)
                cap.release()
                return result
        
        cap.release()
        return None

    def _detect_face_from_frames(self, frames_path, frame_files):
        """Detect face region from extracted frames directory.
        Returns crop params or None if detection fails."""
        # Try first, middle, quarter frames
        n = len(frame_files)
        sample_indices = [0, n // 4, n // 2]
        
        for idx in sample_indices:
            if idx < n:
                fpath = os.path.join(frames_path, frame_files[idx])
                frame = cv2.imread(fpath)
                if frame is not None:
                    return detect_face_region(frame)
        return None

    def _apply_temporal_dropout(self, face_images, body_images):
        """Randomly replace some frames with blank frames during training.
        
        Forces the temporal model to be robust to missing/corrupted frames.
        The same frames are dropped for both face and body streams.
        """
        if self.temporal_dropout <= 0 or self.mode != 'train':
            return face_images, body_images
        
        n = len(face_images)
        num_drop = max(1, int(n * self.temporal_dropout))
        # Don't drop more than 30% of frames
        num_drop = min(num_drop, int(n * 0.3))
        
        drop_indices = random.sample(range(n), num_drop)
        
        for idx in drop_indices:
            # Replace with the previous valid frame (temporal smoothing)
            # rather than black frame, which is less disruptive
            prev_idx = max(0, idx - 1)
            if prev_idx in drop_indices:
                prev_idx = max(0, idx - 2)
            face_images[idx] = face_images[prev_idx].copy()
            body_images[idx] = body_images[prev_idx].copy()
        
        return face_images, body_images

    def _load_frames_from_video(self, video_path, indices, face_crop_params, body_crop_params):
        """Read specific frames from video file with consistent crop params."""
        face_images = []
        body_images = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return face_images, body_images
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for seg_ind in indices:
            p = min(int(seg_ind), total_frames - 1)
            for _ in range(self.duration):
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    face_images.append(self._apply_crop(img_pil, face_crop_params))
                    if body_crop_params is not None:
                        body_images.append(self._apply_crop(img_pil, body_crop_params))
                    else:
                        body_images.append(img_pil)  # Full frame for body
                else:
                    blank = Image.new('RGB', (self.image_size, self.image_size))
                    face_images.append(blank)
                    body_images.append(blank)
                if p < total_frames - 1:
                    p += 1
        cap.release()
        return face_images, body_images

    def __getitem__(self, index):
        clip_dir, label, clip_id_ext = self.samples[index]
        
        # Lazy resolve: check frames/ dir or fallback to video
        if clip_dir not in self.frame_cache:
            frames_path = os.path.join(clip_dir, 'frames')
            video_path = os.path.join(clip_dir, clip_id_ext)
            
            if os.path.isdir(frames_path):
                files = sorted([f for f in os.listdir(frames_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                self.frame_cache[clip_dir] = ('frames', frames_path, files)
            elif os.path.isfile(video_path):
                cap = cv2.VideoCapture(video_path)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                cap.release()
                self.frame_cache[clip_dir] = ('video', video_path, n)
            else:
                try:
                    avi = [f for f in os.listdir(clip_dir) if f.endswith('.avi')]
                    if avi:
                        vp = os.path.join(clip_dir, avi[0])
                        cap = cv2.VideoCapture(vp)
                        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                        cap.release()
                        self.frame_cache[clip_dir] = ('video', vp, n)
                    else:
                        self.frame_cache[clip_dir] = ('empty', '', 0)
                except:
                    self.frame_cache[clip_dir] = ('empty', '', 0)
        
        cache_entry = self.frame_cache[clip_dir]
        source_type = cache_entry[0]
        
        blank_return = (
            torch.zeros(self.num_segments, 3, self.image_size, self.image_size),
            torch.zeros(self.num_segments, 3, self.image_size, self.image_size),
            label
        )
        
        face_images = []
        body_images = []
        
        # Pre-compute crop params ONCE for the entire video (temporal consistency)
        # Use a reference dimension (DAiSEE videos are typically 640x480)
        ref_w, ref_h = 640, 480  # default, will be updated from first frame
        
        # === SOTA: Multi-scale crop ratio selection ===
        crop_ratio = random.choice(self.crop_ratios)
        
        if source_type == 'frames':
            _, frames_path, frame_files = cache_entry
            num_frames = len(frame_files)
            if num_frames <= 0:
                return blank_return
            
            # Get actual frame dimensions from first frame
            try:
                first_img = Image.open(os.path.join(frames_path, frame_files[0]))
                ref_w, ref_h = first_img.size
                first_img.close()
            except:
                pass
            
            # === SOTA: Face detection or adaptive crop ===
            if self.use_face_detection and self.mode == 'train':
                detected = self._detect_face_from_frames(frames_path, frame_files)
                if detected is not None:
                    face_crop_params = detected
                else:
                    face_crop_params = self._get_crop_params(ref_w, ref_h, crop_ratio)
            else:
                face_crop_params = self._get_crop_params(ref_w, ref_h, crop_ratio)
            
            # Body crop params: wider face crop in face_only_mode, or None (full frame)
            if self.face_only_mode:
                body_ratio = random.choice(self.body_crop_ratios)
                body_crop_params = self._get_crop_params(ref_w, ref_h, body_ratio)
            else:
                body_crop_params = None
            
            indices = self._get_indices(num_frames)
            for seg_ind in indices:
                p = min(int(seg_ind), num_frames - 1)
                for _ in range(self.duration):
                    try:
                        img_pil = Image.open(os.path.join(frames_path, frame_files[p])).convert('RGB')
                        face_images.append(self._apply_crop(img_pil, face_crop_params))
                        if body_crop_params is not None:
                            body_images.append(self._apply_crop(img_pil, body_crop_params))
                        else:
                            body_images.append(img_pil)  # Full frame for body
                    except:
                        blank = Image.new('RGB', (self.image_size, self.image_size))
                        face_images.append(blank)
                        body_images.append(blank)
                    if p < num_frames - 1:
                        p += 1
                        
        elif source_type == 'video':
            _, video_path, num_frames = cache_entry
            if num_frames <= 0:
                return blank_return
            
            # Get actual frame dimensions
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ref_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or ref_w
                ref_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or ref_h
                cap.release()
            
            # === SOTA: Face detection or adaptive crop ===
            if self.use_face_detection and self.mode == 'train':
                detected = self._detect_face_from_video(video_path)
                if detected is not None:
                    face_crop_params = detected
                else:
                    face_crop_params = self._get_crop_params(ref_w, ref_h, crop_ratio)
            else:
                face_crop_params = self._get_crop_params(ref_w, ref_h, crop_ratio)
            
            # Body crop params for face_only_mode
            body_crop_params_video = None
            if self.face_only_mode:
                body_ratio = random.choice(self.body_crop_ratios)
                body_crop_params_video = self._get_crop_params(ref_w, ref_h, body_ratio)
            
            indices = self._get_indices(num_frames)
            face_images, body_images = self._load_frames_from_video(video_path, indices, face_crop_params, body_crop_params_video)
        else:
            return blank_return
        
        if not face_images:
            return blank_return

        # === SOTA: Temporal frame dropout ===
        if self.mode == 'train' and self.temporal_dropout > 0:
            face_images, body_images = self._apply_temporal_dropout(face_images, body_images)

        # Apply CONSISTENT ColorJitter during training (same params for all frames)
        if self.mode == 'train' and hasattr(self, '_color_jitter'):
            face_images = self._apply_consistent_color_jitter(face_images)
            body_images = self._apply_consistent_color_jitter(body_images)

        # Apply same group transforms to both streams
        process_face = self.group_transform(face_images)
        process_body = self.group_transform(body_images)
        
        # Reshape to [T, 3, H, W]
        process_face = process_face.view(-1, 3, self.image_size, self.image_size)
        process_body = process_body.view(-1, 3, self.image_size, self.image_size)
        
        # CLIP normalization - vectorized
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        process_face = (process_face - mean) / std
        process_body = (process_body - mean) / std
        
        return process_face, process_body, label

    def __len__(self):
        return len(self.samples)

def daisee_train_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4, max_samples_per_class=0):
    dataset = DAiSEEDataset(root_dir, list_file, mode='train', num_segments=num_segments, duration=duration, image_size=image_size, max_samples_per_class=max_samples_per_class)
    return dataset

def daisee_test_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4, max_samples_per_class=0):
    dataset = DAiSEEDataset(root_dir, list_file, mode='test', num_segments=num_segments, duration=duration, image_size=image_size)
    return dataset


class DAiSEE4DiscreteDataset(DAiSEEDataset):
    """DAiSEE with 4 discrete affective states as classes.
    
    Instead of 4 ordinal levels of Engagement, uses the DOMINANT
    affective state as the class label:
        0: Boredom
        1: Engagement  
        2: Confusion
        3: Frustration
    
    For each video, the column with the highest value becomes the label.
    Ties broken by: Boredom > Confusion > Frustration > Engagement
    (favor minority classes in ties to help imbalance).
    All-zero samples default to Engagement (most neutral).
    """
    STATE_COLS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    # Priority for tie-breaking (higher = preferred in ties)
    TIE_PRIORITY = {'Boredom': 3, 'Confusion': 2, 'Frustration': 1, 'Engagement': 0}
    
    def _make_dataset(self):
        samples = []
        if not os.path.exists(self.annotation_file):
            print(f"Error: Annotation file {self.annotation_file} not found.")
            return samples
            
        print(f"Loading annotations from: {self.annotation_file}")
        try:
            df = pd.read_csv(self.annotation_file)
            df.columns = df.columns.str.strip()  # Fix trailing spaces
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return samples

        if 'Train' in self.annotation_file or self.mode == 'train':
            split_dir = 'Train'
        elif 'Validation' in self.annotation_file or self.mode == 'val':
            split_dir = 'Validation'
        elif 'Test' in self.annotation_file or self.mode == 'test':
            split_dir = 'Test'
        else:
            split_dir = 'Train'

        # Class mapping: column name → label index
        col_to_label = {col: i for i, col in enumerate(self.STATE_COLS)}
        
        print(f"Processing entries for {self.mode} (4-class discrete)...")
        class_counts = {col: 0 for col in self.STATE_COLS}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {self.mode}"):
            clip_id_ext = row['ClipID']
            if not isinstance(clip_id_ext, str):
                continue
            clip_id = os.path.splitext(clip_id_ext)[0]
            subject_id = clip_id[:6]
            
            # Find dominant state
            values = {col: int(row[col]) for col in self.STATE_COLS}
            max_val = max(values.values())
            
            if max_val == 0:
                # All zeros → default to Engagement
                dominant = 'Engagement'
            else:
                # Pick column with highest value, tie-break by priority
                candidates = [col for col, val in values.items() if val == max_val]
                dominant = max(candidates, key=lambda c: self.TIE_PRIORITY[c])
            
            label = col_to_label[dominant]
            class_counts[dominant] += 1
            
            clip_dir = os.path.join(self.root_dir, 'DataSet', split_dir, subject_id, clip_id)
            samples.append((clip_dir, label, clip_id_ext))

        print(f"DAiSEE 4-Discrete ({self.mode}): {len(samples)} samples")
        print(f"  Distribution: {class_counts}")
        return samples
