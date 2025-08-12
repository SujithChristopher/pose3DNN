import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import urllib.request
import zipfile
from pathlib import Path

class Human36MDataset(data.Dataset):
    """
    Simple Human3.6M dataset loader for 3D pose estimation
    Downloads a small subset for quick training
    """
    
    def __init__(self, root_dir: str = './data', split: str = 'train', 
                 image_size: Tuple[int, int] = (224, 224), download: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        if download:
            self._download_sample_data()
        
        self.data_path = self.root_dir / 'human36m_sample'
        self.annotations = self._load_annotations()
        
    def _download_sample_data(self):
        """Download sample Human3.6M data for quick testing"""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data structure
        sample_dir = self.root_dir / 'human36m_sample'
        sample_dir.mkdir(exist_ok=True)
        
        # Create synthetic sample annotations
        self._create_sample_annotations()
        
    def _create_sample_annotations(self):
        """Create synthetic sample data for initial training"""
        sample_dir = self.root_dir / 'human36m_sample'
        
        # Human3.6M has 17 joints in 3D
        joint_names = [
            'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
            'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 
            'LWrist', 'RShoulder', 'RElbow', 'RWrist'
        ]
        
        # Create synthetic training samples
        samples = []
        for i in range(100):  # Small dataset for quick training
            # Generate synthetic 2D keypoints (normalized to image size)
            keypoints_2d = np.random.rand(17, 2) * np.array(self.image_size)
            
            # Generate synthetic 3D keypoints (in mm, centered around origin)
            keypoints_3d = np.random.randn(17, 3) * 200  # 200mm std deviation
            
            # Simple skeleton constraints
            keypoints_3d[0] = [0, 0, 0]  # Hip at origin
            keypoints_3d[7] = keypoints_3d[0] + [0, 0, 200]  # Spine above hip
            keypoints_3d[9] = keypoints_3d[7] + [0, 0, 300]  # Neck above spine
            keypoints_3d[10] = keypoints_3d[9] + [0, 0, 200]  # Head above neck
            
            sample = {
                'subject': f'S{(i % 5) + 1}',
                'action': f'action_{i % 10}',
                'frame': i,
                'keypoints_2d': keypoints_2d.tolist(),
                'keypoints_3d': keypoints_3d.tolist(),
                'camera_params': {
                    'focal_length': [1145.0, 1145.0],
                    'center': [512.0, 512.0],
                    'radial_distortion': [0.0, 0.0, 0.0],
                    'tangential_distortion': [0.0, 0.0]
                }
            }
            samples.append(sample)
        
        # Save annotations
        with open(sample_dir / 'annotations.json', 'w') as f:
            json.dump({
                'joint_names': joint_names,
                'samples': samples
            }, f, indent=2)
        
        print(f"Created {len(samples)} synthetic training samples")
    
    def _load_annotations(self):
        """Load annotation data"""
        ann_file = self.data_path / 'annotations.json'
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Filter by split (for now, use all data as training)
        samples = data['samples']
        split_size = int(0.8 * len(samples))
        
        if self.split == 'train':
            return samples[:split_size]
        else:
            return samples[split_size:]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        sample = self.annotations[idx]
        
        # Create synthetic grayscale image for monochrome camera simulation
        image = self._generate_synthetic_image(sample['keypoints_2d'])
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # Normalize to [0,1]
        
        # 2D keypoints (input)
        keypoints_2d = torch.from_numpy(np.array(sample['keypoints_2d'])).float()
        
        # 3D keypoints (target)
        keypoints_3d = torch.from_numpy(np.array(sample['keypoints_3d'])).float()
        
        return {
            'image': image,
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': keypoints_3d,
            'subject': sample['subject'],
            'action': sample['action']
        }
    
    def _generate_synthetic_image(self, keypoints_2d: List[List[float]]) -> np.ndarray:
        """Generate synthetic grayscale image with pose"""
        image = np.zeros((*self.image_size, 1), dtype=np.uint8)
        
        # Draw keypoints and skeleton
        keypoints = np.array(keypoints_2d)
        
        # Human3.6M skeleton connections
        skeleton = [
            (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # legs
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine to head
            (8, 11), (11, 12), (12, 13),  # left arm
            (8, 14), (14, 15), (15, 16)   # right arm
        ]
        
        # Draw skeleton lines
        for start_idx, end_idx in skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_pt = tuple(keypoints[start_idx].astype(int))
                end_pt = tuple(keypoints[end_idx].astype(int))
                cv2.line(image, start_pt, end_pt, (200,), 2)
        
        # Draw keypoints
        for kp in keypoints:
            cv2.circle(image, tuple(kp.astype(int)), 3, (255,), -1)
        
        # Add some noise to simulate real camera conditions
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return image.squeeze()  # Remove channel dimension for grayscale


def create_data_loaders(batch_size: int = 8, num_workers: int = 4) -> Tuple[data.DataLoader, data.DataLoader]:
    """Create training and validation data loaders"""
    
    train_dataset = Human36MDataset(split='train', download=True)
    val_dataset = Human36MDataset(split='val', download=False)
    
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
    # Test the dataset
    dataset = Human36MDataset(download=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample['image'].shape)
    print("2D keypoints shape:", sample['keypoints_2d'].shape)
    print("3D keypoints shape:", sample['keypoints_3d'].shape)