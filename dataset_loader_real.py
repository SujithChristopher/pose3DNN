#!/usr/bin/env python3
"""
Enhanced dataset loader supporting both synthetic and real Human3.6M data
"""

import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random

class Human36MRealDataset(data.Dataset):
    """
    Real Human3.6M dataset loader
    Loads processed Human3.6M poses and creates training samples
    """
    
    def __init__(self, 
                 data_file: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (224, 224),
                 augment: bool = True):
        
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Load data
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        with open(data_file, 'r') as f:
            dataset = json.load(f)
        
        self.metadata = dataset['metadata']
        self.data = dataset['data']
        self.joint_names = self.metadata['joint_names']
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
        print(f"Subjects: {self.metadata.get('subjects', 'Unknown')}")
        print(f"Dataset: {self.metadata.get('dataset', 'Unknown')}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get poses
        pose_2d = np.array(sample['pose_2d'], dtype=np.float32)  # (17, 2)
        pose_3d = np.array(sample['pose_3d'], dtype=np.float32)  # (17, 3)
        
        # Normalize 2D poses to image coordinates
        pose_2d_normalized = self._normalize_2d_pose(pose_2d)
        
        # Center 3D pose (make root-relative)
        pose_3d_centered = pose_3d - pose_3d[0:1]  # Subtract hip position
        
        # Data augmentation
        if self.augment and self.split == 'train':
            pose_2d_normalized, pose_3d_centered = self._augment_poses(
                pose_2d_normalized, pose_3d_centered
            )
        
        # Create synthetic image based on 2D pose
        image = self._create_pose_image(pose_2d_normalized)
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # (1, H, W)
        pose_2d_tensor = torch.from_numpy(pose_2d_normalized).float()  # (17, 2)
        pose_3d_tensor = torch.from_numpy(pose_3d_centered).float()    # (17, 3)
        
        return {
            'image': image,
            'keypoints_2d': pose_2d_tensor,
            'keypoints_3d': pose_3d_tensor,
            'subject': sample['subject'],
            'action': sample['action'],
            'camera': sample.get('camera', 1),
            'frame': sample.get('frame', 0)
        }
    
    def _normalize_2d_pose(self, pose_2d: np.ndarray) -> np.ndarray:
        """Normalize 2D pose to image coordinates"""
        # Human3.6M poses are in different coordinate system
        # Normalize to [0, 1] range, then scale to image size
        
        # Remove invalid points (set to image center if needed)
        valid_mask = ~np.any(np.isnan(pose_2d) | np.isinf(pose_2d), axis=1)
        
        if not np.any(valid_mask):
            # If no valid points, create dummy pose
            pose_2d_norm = np.random.rand(17, 2)
            pose_2d_norm[:, 0] *= self.image_size[1]  # width
            pose_2d_norm[:, 1] *= self.image_size[0]  # height
            return pose_2d_norm
        
        # Get bounding box of valid points
        valid_poses = pose_2d[valid_mask]
        min_x, min_y = valid_poses.min(axis=0)
        max_x, max_y = valid_poses.max(axis=0)
        
        # Add padding
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        scale = max(max_x - min_x, max_y - min_y) * 1.25
        
        # Normalize to image coordinates
        pose_2d_norm = pose_2d.copy()
        pose_2d_norm[:, 0] = (pose_2d[:, 0] - center_x) / scale * self.image_size[1] * 0.8 + self.image_size[1] / 2
        pose_2d_norm[:, 1] = (pose_2d[:, 1] - center_y) / scale * self.image_size[0] * 0.8 + self.image_size[0] / 2
        
        # Clamp to image bounds
        pose_2d_norm[:, 0] = np.clip(pose_2d_norm[:, 0], 0, self.image_size[1] - 1)
        pose_2d_norm[:, 1] = np.clip(pose_2d_norm[:, 1], 0, self.image_size[0] - 1)
        
        return pose_2d_norm
    
    def _augment_poses(self, pose_2d: np.ndarray, pose_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to poses"""
        
        # 2D augmentations
        if random.random() < 0.5:  # Horizontal flip
            pose_2d[:, 0] = self.image_size[1] - 1 - pose_2d[:, 0]
            pose_3d[:, 0] = -pose_3d[:, 0]
            
            # Swap left/right joints
            left_right_pairs = [
                (1, 4),   # RHip <-> LHip
                (2, 5),   # RKnee <-> LKnee  
                (3, 6),   # RFoot <-> LFoot
                (11, 14), # LShoulder <-> RShoulder
                (12, 15), # LElbow <-> RElbow
                (13, 16), # LWrist <-> RWrist
            ]
            
            for left_idx, right_idx in left_right_pairs:
                # Swap 2D points
                pose_2d[[left_idx, right_idx]] = pose_2d[[right_idx, left_idx]]
                # Swap 3D points
                pose_3d[[left_idx, right_idx]] = pose_3d[[right_idx, left_idx]]
        
        # 2D translation
        if random.random() < 0.3:
            tx = random.uniform(-20, 20)
            ty = random.uniform(-20, 20)
            pose_2d[:, 0] = np.clip(pose_2d[:, 0] + tx, 0, self.image_size[1] - 1)
            pose_2d[:, 1] = np.clip(pose_2d[:, 1] + ty, 0, self.image_size[0] - 1)
        
        # 2D scale
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            center_x, center_y = self.image_size[1] / 2, self.image_size[0] / 2
            pose_2d[:, 0] = (pose_2d[:, 0] - center_x) * scale + center_x
            pose_2d[:, 1] = (pose_2d[:, 1] - center_y) * scale + center_y
            pose_2d[:, 0] = np.clip(pose_2d[:, 0], 0, self.image_size[1] - 1)
            pose_2d[:, 1] = np.clip(pose_2d[:, 1], 0, self.image_size[0] - 1)
        
        # 3D rotation around Y-axis (global rotation)
        if random.random() < 0.5:
            angle = random.uniform(-30, 30) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_y = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            pose_3d = pose_3d @ rotation_y.T
        
        # Add 3D noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, pose_3d.shape)  # 5mm std
            pose_3d += noise
        
        return pose_2d, pose_3d
    
    def _create_pose_image(self, pose_2d: np.ndarray) -> np.ndarray:
        """Create synthetic image from 2D pose (similar to original)"""
        image = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        
        # Add background noise
        noise = np.random.normal(20, 10, image.shape).astype(np.int16)
        image = np.clip(noise, 0, 50).astype(np.uint8)
        
        # Human3.6M skeleton connections
        skeleton = [
            (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # legs
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine to head
            (8, 11), (11, 12), (12, 13),  # left arm
            (8, 14), (14, 15), (15, 16)   # right arm
        ]
        
        # Draw skeleton lines
        for start_idx, end_idx in skeleton:
            if start_idx < len(pose_2d) and end_idx < len(pose_2d):
                start_pt = tuple(pose_2d[start_idx].astype(int))
                end_pt = tuple(pose_2d[end_idx].astype(int))
                cv2.line(image, start_pt, end_pt, (180,), 3)
        
        # Draw joints
        for i, kp in enumerate(pose_2d):
            pos = tuple(kp.astype(int))
            radius = 4 if i == 0 else 3  # Hip larger
            cv2.circle(image, pos, radius, (255,), -1)
        
        # Apply blur
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return image

class HybridDataset(data.Dataset):
    """
    Hybrid dataset that combines synthetic and real Human3.6M data
    """
    
    def __init__(self,
                 synthetic_ratio: float = 0.3,
                 real_data_file: Optional[str] = None,
                 **kwargs):
        
        self.synthetic_ratio = synthetic_ratio
        self.datasets = []
        
        # Add synthetic dataset
        from dataset_loader import Human36MDataset
        synthetic_dataset = Human36MDataset(**kwargs)
        self.datasets.append(('synthetic', synthetic_dataset))
        
        # Add real dataset if available
        if real_data_file and os.path.exists(real_data_file):
            real_dataset = Human36MRealDataset(real_data_file, **kwargs)
            self.datasets.append(('real', real_dataset))
            print(f"Hybrid dataset: {len(synthetic_dataset)} synthetic + {len(real_dataset)} real samples")
        else:
            print(f"Hybrid dataset: {len(synthetic_dataset)} synthetic samples only")
            print("To add real data, download Human3.6M and process it first")
    
    def __len__(self):
        return sum(len(dataset) for _, dataset in self.datasets)
    
    def __getitem__(self, idx):
        # Choose dataset based on ratio
        if len(self.datasets) > 1:
            if random.random() < self.synthetic_ratio:
                dataset_name, dataset = self.datasets[0]  # synthetic
                sample_idx = idx % len(dataset)
            else:
                dataset_name, dataset = self.datasets[1]  # real
                sample_idx = idx % len(dataset)
        else:
            dataset_name, dataset = self.datasets[0]
            sample_idx = idx % len(dataset)
        
        sample = dataset[sample_idx]
        sample['data_source'] = dataset_name
        return sample

def create_human36m_loaders(data_type: str = 'synthetic',
                           real_train_file: Optional[str] = None,
                           real_test_file: Optional[str] = None,
                           batch_size: int = 8,
                           num_workers: int = 4) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Create data loaders for different dataset types
    
    Args:
        data_type: 'synthetic', 'real', or 'hybrid'
        real_train_file: Path to processed Human3.6M train data
        real_test_file: Path to processed Human3.6M test data
    """
    
    if data_type == 'synthetic':
        # Use original synthetic dataset
        from dataset_loader import create_data_loaders
        return create_data_loaders(batch_size, num_workers)
    
    elif data_type == 'real':
        if not real_train_file or not os.path.exists(real_train_file):
            raise FileNotFoundError(f"Real training data not found: {real_train_file}")
        
        train_dataset = Human36MRealDataset(real_train_file, split='train', augment=True)
        
        if real_test_file and os.path.exists(real_test_file):
            val_dataset = Human36MRealDataset(real_test_file, split='test', augment=False)
        else:
            # Split training data for validation
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = data.random_split(train_dataset, [train_size, val_size])
    
    elif data_type == 'hybrid':
        train_dataset = HybridDataset(
            synthetic_ratio=0.3,
            real_data_file=real_train_file,
            split='train',
            augment=True
        )
        
        if real_test_file and os.path.exists(real_test_file):
            val_dataset = Human36MRealDataset(real_test_file, split='test', augment=False)
        else:
            # Use synthetic validation
            from dataset_loader import Human36MDataset
            val_dataset = Human36MDataset(split='val')
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset loaders
    print("Testing dataset loaders...")
    
    # Test synthetic
    print("\n1. Testing synthetic dataset:")
    train_loader, val_loader = create_human36m_loaders('synthetic', batch_size=4)
    batch = next(iter(train_loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Batch shapes: {batch['image'].shape}, {batch['keypoints_2d'].shape}, {batch['keypoints_3d'].shape}")
    
    # Test real (if available)
    real_file = "./data/human36m/processed/sample_train.json"
    if os.path.exists(real_file):
        print(f"\n2. Testing real dataset:")
        train_loader, val_loader = create_human36m_loaders('real', real_file, batch_size=4)
        batch = next(iter(train_loader))
        print(f"   Batch keys: {batch.keys()}")
        print(f"   Data source: Real Human3.6M")
        
        print(f"\n3. Testing hybrid dataset:")
        train_loader, val_loader = create_human36m_loaders('hybrid', real_file, batch_size=4)
        batch = next(iter(train_loader))
        print(f"   Data source: {batch['data_source']}")
    else:
        print(f"\n2. Real dataset not available at: {real_file}")
        print(f"   Download and process Human3.6M first")
    
    print(f"\n✅ Dataset loader tests completed!")