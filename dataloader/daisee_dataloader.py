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
from dataloader.video_transform import GroupResize, Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class DAiSEEDataset(data.Dataset):
    def __init__(self, root_dir, annotation_file, mode='train', num_segments=16, duration=1, image_size=224):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.mode = mode
        self.num_segments = num_segments
        self.duration = duration
        self.image_size = image_size
        self.label_col = 'Engagement' 
        self.frame_cache = {} 
        
        self.samples = self._make_dataset()
        
        # Group transforms for face stream
        if mode == 'train':
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
        
        # Motion stream only needs resize + stack (no flip — motion direction matters)
        self.motion_transform = transforms.Compose([
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
            clip_dir = os.path.join(self.root_dir, 'DataSet', split_dir, subject_id, clip_id)
            samples.append((clip_dir, label, clip_id_ext))

        print(f"DAiSEE ({self.mode}): Loaded {len(samples)} samples.")
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

    def _center_crop_face(self, img_pil, crop_ratio=0.4):
        """Tighter crop to focus on face (40% of frame)."""
        w, h = img_pil.size
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        return img_pil.crop((left, top, left + crop_w, top + crop_h))

    def _compute_motion_frames(self, pil_images):
        """Compute temporal difference (motion) between consecutive frames.
        Returns list of PIL images showing motion intensity."""
        motion_frames = []
        for i in range(len(pil_images)):
            if i == 0:
                # First frame: diff with itself = gray image
                motion_frames.append(Image.new('RGB', pil_images[0].size, (128, 128, 128)))
            else:
                # Convert to numpy, compute absolute difference
                curr = np.array(pil_images[i]).astype(np.float32)
                prev = np.array(pil_images[i-1]).astype(np.float32)
                diff = np.abs(curr - prev)
                # Amplify motion signal (x3) and clip to 0-255
                diff = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
                motion_frames.append(Image.fromarray(diff))
        return motion_frames

    def _load_frames_from_video(self, video_path, indices):
        """Read specific frames from video file."""
        raw_images = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for seg_ind in indices:
            p = min(int(seg_ind), total_frames - 1)
            for _ in range(self.duration):
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_images.append(Image.fromarray(frame_rgb))
                else:
                    raw_images.append(Image.new('RGB', (self.image_size, self.image_size)))
                if p < total_frames - 1:
                    p += 1
        cap.release()
        
        # Create face crops and motion frames from raw images
        face_images = [self._center_crop_face(img, 0.4) for img in raw_images]
        motion_images = self._compute_motion_frames(raw_images)
        return face_images, motion_images

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
        
        if source_type == 'frames':
            _, frames_path, frame_files = cache_entry
            num_frames = len(frame_files)
            if num_frames <= 0:
                return blank_return
            
            indices = self._get_indices(num_frames)
            raw_images = []
            for seg_ind in indices:
                p = min(int(seg_ind), num_frames - 1)
                for _ in range(self.duration):
                    try:
                        img_pil = Image.open(os.path.join(frames_path, frame_files[p])).convert('RGB')
                        raw_images.append(img_pil)
                    except:
                        raw_images.append(Image.new('RGB', (self.image_size, self.image_size)))
                    if p < num_frames - 1:
                        p += 1
            
            face_images = [self._center_crop_face(img, 0.4) for img in raw_images]
            motion_images = self._compute_motion_frames(raw_images)
                        
        elif source_type == 'video':
            _, video_path, num_frames = cache_entry
            if num_frames <= 0:
                return blank_return
            indices = self._get_indices(num_frames)
            face_images, motion_images = self._load_frames_from_video(video_path, indices)
        else:
            return blank_return
        
        if not face_images:
            return blank_return

        # Apply transforms
        process_face = self.group_transform(face_images)       # Face: with augmentation
        process_motion = self.motion_transform(motion_images)  # Motion: no flip
        
        # Reshape to [T, 3, H, W]
        process_face = process_face.view(-1, 3, self.image_size, self.image_size)
        process_motion = process_motion.view(-1, 3, self.image_size, self.image_size)
        
        # CLIP normalization for face stream
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        process_face = (process_face - mean) / std
        
        # Simple normalization for motion stream (0-1 range, then standard norm)
        process_motion = (process_motion - 0.5) / 0.5  # Scale to [-1, 1]
        
        return process_face, process_motion, label

    def __len__(self):
        return len(self.samples)

def daisee_train_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    dataset = DAiSEEDataset(root_dir, list_file, mode='train', num_segments=num_segments, duration=duration, image_size=image_size)
    return dataset

def daisee_test_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    dataset = DAiSEEDataset(root_dir, list_file, mode='test', num_segments=num_segments, duration=duration, image_size=image_size)
    return dataset
