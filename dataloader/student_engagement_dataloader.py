"""
Student Engagement Dataset Dataloader
Dataset: https://www.kaggle.com/datasets/joyee19/studentengagement
Structure:
    Student-engagement/
        Engaged/
            confused/
            engaged/
            frustrated/
        Not Engaged/
            Looking away/
            bored/
            drowsy/

Binary classification: Engaged (0) vs Not Engaged (1)
Static images → replicated to num_segments for temporal module compatibility.
"""
import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from dataloader.video_transform import GroupResize, Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class StudentEngagementDataset(data.Dataset):
    def __init__(self, root_dir, mode='train', num_segments=8, image_size=224, val_ratio=0.15, test_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.image_size = image_size
        
        # Class mapping: auto-detect folder names
        self.class_map = self._detect_classes(root_dir)
        
        # Build full dataset then split
        all_samples = self._scan_dataset()
        self.samples = self._split_dataset(all_samples, val_ratio, test_ratio, seed)
        
        # Augmentation
        if mode == 'train':
            self._color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        
        # CLIP normalization
        self.normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    
    def _detect_classes(self, root_dir):
        """Auto-detect class folder names."""
        class_map = {}
        engaged_variants = ['engaged', 'Engaged', 'ENGAGED']
        not_engaged_variants = ['not engaged', 'Not Engaged', 'Not engaged', 'NotEngaged', 
                                'not_engaged', 'NOT ENGAGED', 'Notengaged']
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if not os.path.isdir(item_path):
                continue
            item_lower = item.lower().strip()
            if item_lower.startswith('not') and 'engag' in item_lower:
                class_map[item] = 1  # Not Engaged
            elif 'engag' in item_lower:
                class_map[item] = 0  # Engaged
        
        if not class_map:
            # Fallback: use all subdirectories as classes
            dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            for i, d in enumerate(dirs):
                class_map[d] = i
        
        print(f"Auto-detected class mapping: {class_map}")
        return class_map
    
    def _scan_dataset(self):
        """Scan folder structure and collect all (path, label) pairs."""
        samples = []
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            # Scan for images in class_dir and all subdirectories
            for dirpath, dirnames, filenames in os.walk(class_dir):
                for fname in filenames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        fpath = os.path.join(dirpath, fname)
                        samples.append((fpath, label))
        print(f"Total scanned: {len(samples)} images")
        return samples
    
    def _split_dataset(self, all_samples, val_ratio, test_ratio, seed):
        """Deterministic stratified split."""
        rng = random.Random(seed)
        
        # Group by label
        by_label = {}
        for path, label in all_samples:
            by_label.setdefault(label, []).append((path, label))
        
        train, val, test = [], [], []
        for label, items in by_label.items():
            rng.shuffle(items)
            n = len(items)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            test.extend(items[:n_test])
            val.extend(items[n_test:n_test + n_val])
            train.extend(items[n_test + n_val:])
        
        if self.mode == 'train':
            result = train
        elif self.mode == 'val':
            result = val
        else:
            result = test
        
        # Print stats
        print(f"StudentEngagement ({self.mode}): {len(result)} samples")
        label_counts = {}
        for _, l in result:
            label_counts[l] = label_counts.get(l, 0) + 1
        print(f"  Distribution: {label_counts}")
        
        return result
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        
        # Apply color jitter during training (on PIL image)
        if self.mode == 'train' and hasattr(self, '_color_jitter'):
            img = self._color_jitter(img)
        
        # Apply transform (crop/resize + to tensor)
        img_tensor = self.transform(img)  # [3, H, W]
        
        # CLIP normalize
        img_tensor = self.normalize(img_tensor)
        
        # Replicate to num_segments for temporal module: [T, 3, H, W]
        face_frames = img_tensor.unsqueeze(0).repeat(self.num_segments, 1, 1, 1)
        body_frames = face_frames.clone()
        
        return face_frames, body_frames, label
    
    def __len__(self):
        return len(self.samples)


class StudentEngagement6Dataset(StudentEngagementDataset):
    """6-class version: confused(0), engaged(1), frustrated(2), looking_away(3), bored(4), drowsy(5)"""
    
    # Fixed subclass → label mapping
    SUBCLASS_MAP = {
        'confused': 0, 'Confused': 0,
        'engaged': 1, 'Engaged': 1,
        'frustrated': 2, 'Frustrated': 2,
        'looking away': 3, 'Looking away': 3, 'Looking Away': 3, 'lookingaway': 3,
        'bored': 4, 'Bored': 4,
        'drowsy': 5, 'Drowsy': 5,
    }
    
    def _detect_classes(self, root_dir):
        """Override: find subclass folders at any depth."""
        class_map = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            basename = os.path.basename(dirpath)
            for name, label in self.SUBCLASS_MAP.items():
                if basename.lower().strip() == name.lower():
                    # Use absolute path as key
                    class_map[dirpath] = label
                    break
        
        if not class_map:
            # Debug: show what we found
            print(f"ERROR: No subclass folders found. Directory tree:")
            for dirpath, dirnames, _ in os.walk(root_dir):
                depth = dirpath.replace(root_dir, '').count(os.sep)
                if depth <= 3:
                    print(f"  {'  ' * depth}{os.path.basename(dirpath)}/")
        
        print(f"6-class mapping ({len(class_map)} folders): { {os.path.basename(k): v for k, v in class_map.items()} }")
        return class_map
    
    def _scan_dataset(self):
        """Override: class_map keys are absolute paths to subclass folders."""
        samples = []
        for folder_path, label in self.class_map.items():
            if not os.path.isdir(folder_path):
                print(f"Warning: Directory not found: {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    fpath = os.path.join(folder_path, fname)
                    samples.append((fpath, label))
        print(f"Total scanned (6-class): {len(samples)} images")
        return samples
