import os
import cv2
import pandas as pd
import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloader.video_transform import GroupResize, Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

class DAiSEEDataset(data.Dataset):
    def __init__(self, root_dir, annotation_file, mode='train', num_segments=16, duration=1, image_size=224):
        """
        Args:
            root_dir (str): Path to the DAiSEE dataset root (containing DataSet and Labels).
            annotation_file (str): Path to the CSV label file.
            mode (str): 'train', 'val', or 'test'.
            num_segments (int): Number of segments to sample.
            duration (int): Number of frames per segment.
            image_size (int): Target image size.
        """
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.mode = mode
        self.num_segments = num_segments
        self.duration = duration
        self.image_size = image_size
        
        # Engagement label index in CSV: ClipID, Boredom, Engagement, Confusion, Frustration
        # Engagement is index 2.
        self.label_col = 'Engagement' 
        
        self.samples = self._make_dataset()
        
        if mode == 'train':
            self.transform = transforms.Compose([
                GroupResize(image_size),
                GroupRandomHorizontalFlip(),
                Stack(),
                ToTorchFormatTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                GroupResize(image_size),
                Stack(),
                ToTorchFormatTensor(),
            ])

    def _make_dataset(self):
        samples = []
        if not os.path.exists(self.annotation_file):
            print(f"Error: Annotation file {self.annotation_file} not found.")
            return samples
            
        df = pd.read_csv(self.annotation_file)
        # Columns: ClipID, Boredom, Engagement, Confusion, Frustration
        # ClipID example: 1100011002.avi
        
        # Determine split directory based on mode or infer from path?
        # The user provided structure: DAiSEE/DataSet/Train, DAiSEE/DataSet/Validation, DAiSEE/DataSet/Test
        if 'Train' in self.annotation_file or self.mode == 'train':
            split_dir = 'Train'
        elif 'Validation' in self.annotation_file or self.mode == 'val':
            split_dir = 'Validation'
        elif 'Test' in self.annotation_file or self.mode == 'test':
            split_dir = 'Test'
        else:
            # Fallback based on mode
            if self.mode == 'train': split_dir = 'Train'
            elif self.mode == 'val': split_dir = 'Validation'
            else: split_dir = 'Test'

        for _, row in df.iterrows():
            clip_id_ext = row['ClipID'] # 1100011002.avi
            if not isinstance(clip_id_ext, str):
                continue
                
            clip_id = os.path.splitext(clip_id_ext)[0] # 1100011002
            subject_id = clip_id[:6] # 110001
            
            label = int(row[self.label_col])
            
            # Construct path: DAiSEE/DataSet/{split}/{subject_id}/{clip_id}/{clip_id_ext}
            # Note: The directory structure observed was DAiSEE/DataSet/Train/110001/1100011002/1100011002.avi
            # OR DAiSEE/DataSet/Train/110001/1100011002/frames/ (we saw frames/ dir too)
            
            # Let's check for the video file first
            rel_path = os.path.join('DataSet', split_dir, subject_id, clip_id, clip_id_ext)
            full_path = os.path.join(self.root_dir, rel_path)
            
            if not os.path.exists(full_path):
                 # Try finding frames folder if video doesn't exist
                 rel_path_frames = os.path.join('DataSet', split_dir, subject_id, clip_id, 'frames')
                 full_path_frames = os.path.join(self.root_dir, rel_path_frames)
                 if os.path.exists(full_path_frames):
                     samples.append((full_path_frames, label, 'frames'))
                 else:
                     # print(f"Warning: File not found {full_path}")
                     pass
            else:
                samples.append((full_path, label, 'video'))

        print(f"DAiSEE ({self.mode}): Loaded {len(samples)} samples.")
        return samples

    def _get_indices(self, num_frames):
        if self.mode == 'train':
            # Random sampling for training
            average_duration = (num_frames - self.duration + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)
            elif num_frames > self.num_segments:
                offsets = np.sort(np.random.randint(num_frames - self.duration + 1, size=self.num_segments))
            else:
                offsets = np.pad(np.array(list(range(num_frames))), (0, self.num_segments - num_frames), "edge")
        else:
            # Uniform sampling for test/val
            if num_frames > self.num_segments + self.duration - 1:
                tick = (num_frames - self.duration + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.pad(np.array(list(range(num_frames))), (0, self.num_segments - num_frames), "edge")
        return offsets

    def __getitem__(self, index):
        path, label, type_ = self.samples[index]
        
        images = []
        if type_ == 'video':
            cap = cv2.VideoCapture(path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if num_frames <= 0:
                # Handle broken video
                images = [Image.new('RGB', (self.image_size, self.image_size))] * (self.num_segments * self.duration)
            else:
                indices = self._get_indices(num_frames)
                for seg_ind in indices:
                    p = int(seg_ind)
                    for _ in range(self.duration):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(frame)
                        else:
                            img_pil = Image.new('RGB', (self.image_size, self.image_size))
                        images.append(img_pil)
                        if p < num_frames - 1:
                            p += 1
            cap.release()
        
        elif type_ == 'frames':
            # Assume frames are sorted
            frame_files = sorted(os.listdir(path))
            num_frames = len(frame_files)
            if num_frames <= 0:
                 images = [Image.new('RGB', (self.image_size, self.image_size))] * (self.num_segments * self.duration)
            else:
                indices = self._get_indices(num_frames)
                for seg_ind in indices:
                    p = int(seg_ind)
                    p = min(p, num_frames - 1)
                    for _ in range(self.duration):
                        img_path = os.path.join(path, frame_files[p])
                        try:
                            img_pil = Image.open(img_path).convert('RGB')
                        except:
                            img_pil = Image.new('RGB', (self.image_size, self.image_size))
                        images.append(img_pil)
                        if p < num_frames - 1:
                            p += 1

        if not images:
             images = [Image.new('RGB', (self.image_size, self.image_size))] * (self.num_segments * self.duration)

        # Apply transforms
        # transform expects list of PIL images, returns (C*T, H, W)
        # But we need to separate Face and Body?
        # The model expects t_face, t_body.
        # DAiSEE is mostly face/upper body. We can use the same image for both or crop.
        # Since we don't have bbox, we'll use the full frame (resized) for both.
        # Optionally we could center crop for 'face'.
        
        process_body = self.transform(images) # (3*T, H, W) where T=num_segments*duration
        
        # For face, let's just duplicate body for now as we don't have bbox
        process_face = process_body.clone()
        
        # Reshape to (T, 3, H, W) -> No, video_dataloader reshapes to (Batch, 3, H, W) effectively if T=1
        # Check video_dataloader return shape:
        # process_body = process_body.view(-1, 3, self.image_size, self.image_size) -> (T, 3, H, W)
        
        process_body = process_body.view(-1, 3, self.image_size, self.image_size)
        process_face = process_face.view(-1, 3, self.image_size, self.image_size)
        
        return process_face, process_body, label

    def __len__(self):
        return len(self.samples)

def daisee_train_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    dataset = DAiSEEDataset(root_dir, list_file, mode='train', num_segments=num_segments, duration=duration, image_size=image_size)
    return dataset

def daisee_test_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    dataset = DAiSEEDataset(root_dir, list_file, mode='test', num_segments=num_segments, duration=duration, image_size=image_size)
    return dataset
