#!/usr/bin/env python3
"""
Human3.6M Dataset Downloader and Processor
Downloads and prepares Human3.6M data for 3D pose estimation training
"""

import os
import json
import requests
from pathlib import Path
import subprocess
import zipfile
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import argparse

class Human36MDownloader:
    """Download and process Human3.6M dataset"""
    
    def __init__(self, root_dir="./data/human36m"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Human3.6M joint names and indices
        self.h36m_joints = [
            'Hip',           # 0
            'RHip',          # 1  
            'RKnee',         # 2
            'RFoot',         # 3
            'LHip',          # 4
            'LKnee',         # 5
            'LFoot',         # 6
            'Spine',         # 7
            'Thorax',        # 8
            'Neck/Nose',     # 9
            'Head',          # 10
            'LShoulder',     # 11
            'LElbow',        # 12
            'LWrist',        # 13
            'RShoulder',     # 14
            'RElbow',        # 15
            'RWrist'         # 16
        ]
        
        # Available subjects and actions
        self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        self.train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        self.test_subjects = ['S9', 'S11']
        
        self.actions = [
            'Directions', 'Discussion', 'Eating', 'Greeting',
            'Phoning', 'Posing', 'Purchases', 'Sitting',
            'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting',
            'Walking', 'WalkingDog', 'WalkTogether'
        ]
    
    def print_download_instructions(self):
        """Print instructions for manual download"""
        print("📥 Human3.6M Dataset Download Instructions")
        print("=" * 50)
        print()
        print("Unfortunately, Human3.6M requires manual registration and download.")
        print("Here's how to get the dataset:")
        print()
        print("1. 🌐 Visit: http://vision.imar.ro/human3.6m/")
        print("2. 📝 Register for an account (academic use only)")
        print("3. 📧 Wait for email approval (usually 1-2 days)")
        print("4. 💾 Download these files to ./data/human36m/:")
        print()
        print("   RECOMMENDED DOWNLOAD (smaller, processed):")
        print("   - Poses_D3_Positions_mono_universal.tgz  (~500MB)")
        print("   - Poses_D2_Positions_mono_universal.tgz  (~200MB)")
        print()
        print("   OPTIONAL (for images, much larger):")
        print("   - Videos.tgz (S1, S5, S6, S7, S8, S9, S11)  (~100GB+)")
        print()
        print("5. 🔧 After download, run:")
        print("   python download_human36m.py --process")
        print()
        print("📂 Expected directory structure:")
        print("./data/human36m/")
        print("├── Poses_D3_Positions_mono_universal.tgz")
        print("├── Poses_D2_Positions_mono_universal.tgz")
        print("└── Videos/ (optional)")
        print()
    
    def check_files(self):
        """Check if required files are downloaded"""
        required_files = [
            "Poses_D3_Positions_mono_universal.tgz",
            "Poses_D2_Positions_mono_universal.tgz"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.root_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        else:
            print("✅ All required files found!")
            return True
    
    def extract_files(self):
        """Extract downloaded tgz files"""
        print("📦 Extracting dataset files...")
        
        tgz_files = list(self.root_dir.glob("*.tgz"))
        
        for tgz_file in tgz_files:
            print(f"  Extracting {tgz_file.name}...")
            
            # Extract using tar
            cmd = f"tar -xzf {tgz_file} -C {self.root_dir}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"    ✅ Extracted {tgz_file.name}")
            else:
                print(f"    ❌ Failed to extract {tgz_file.name}")
                print(f"    Error: {result.stderr}")
    
    def load_poses(self):
        """Load 2D and 3D poses from extracted files"""
        print("📊 Loading pose data...")
        
        poses_2d_dir = self.root_dir / "Poses_D2_Positions_mono_universal"
        poses_3d_dir = self.root_dir / "Poses_D3_Positions_mono_universal"
        
        if not poses_2d_dir.exists() or not poses_3d_dir.exists():
            print("❌ Pose directories not found. Extract files first.")
            return None, None
        
        poses_2d = {}
        poses_3d = {}
        
        # Load 2D poses
        for subject in tqdm(self.subjects, desc="Loading 2D poses"):
            poses_2d[subject] = {}
            for action in self.actions:
                for cam in [1, 2, 3, 4]:  # 4 cameras
                    key = f"{subject}_{action}_{cam}"
                    pose_file = poses_2d_dir / f"D2_Positions_mono_universal_{subject}_{action}_{cam}.h5"
                    
                    if pose_file.exists():
                        with h5py.File(pose_file, 'r') as f:
                            poses_2d[subject][f"{action}_{cam}"] = f['Poses'][:]
        
        # Load 3D poses
        for subject in tqdm(self.subjects, desc="Loading 3D poses"):
            poses_3d[subject] = {}
            for action in self.actions:
                key = f"{subject}_{action}"
                pose_file = poses_3d_dir / f"D3_Positions_mono_universal_{subject}_{action}.h5"
                
                if pose_file.exists():
                    with h5py.File(pose_file, 'r') as f:
                        poses_3d[subject][action] = f['Poses'][:]
        
        print(f"✅ Loaded poses for {len(poses_2d)} subjects")
        return poses_2d, poses_3d
    
    def convert_to_training_format(self, poses_2d, poses_3d):
        """Convert poses to format suitable for training"""
        print("🔄 Converting to training format...")
        
        train_data = []
        test_data = []
        
        for subject in self.subjects:
            is_train = subject in self.train_subjects
            
            for action in self.actions:
                if action not in poses_3d[subject]:
                    continue
                
                pose_3d_seq = poses_3d[subject][action]
                
                # Process each camera view
                for cam in [1, 2, 3, 4]:
                    action_cam = f"{action}_{cam}"
                    if action_cam not in poses_2d[subject]:
                        continue
                    
                    pose_2d_seq = poses_2d[subject][action_cam]
                    
                    # Ensure sequences have same length
                    min_len = min(len(pose_2d_seq), len(pose_3d_seq))
                    
                    for frame_idx in range(min_len):
                        # Reshape poses
                        pose_2d = pose_2d_seq[frame_idx].reshape(-1, 2)[:17]  # 17 joints
                        pose_3d = pose_3d_seq[frame_idx].reshape(-1, 3)[:17]  # 17 joints
                        
                        # Create sample
                        sample = {
                            'subject': subject,
                            'action': action,
                            'camera': cam,
                            'frame': frame_idx,
                            'pose_2d': pose_2d.tolist(),
                            'pose_3d': pose_3d.tolist(),
                            'joint_names': self.h36m_joints
                        }
                        
                        if is_train:
                            train_data.append(sample)
                        else:
                            test_data.append(sample)
        
        print(f"✅ Created {len(train_data)} training samples")
        print(f"✅ Created {len(test_data)} test samples")
        
        return train_data, test_data
    
    def save_processed_data(self, train_data, test_data):
        """Save processed data to JSON files"""
        print("💾 Saving processed dataset...")
        
        processed_dir = self.root_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Save training data
        train_file = processed_dir / "train.json"
        with open(train_file, 'w') as f:
            json.dump({
                'metadata': {
                    'dataset': 'Human3.6M',
                    'joints': self.h36m_joints,
                    'subjects': self.train_subjects,
                    'num_samples': len(train_data)
                },
                'data': train_data
            }, f)
        
        # Save test data
        test_file = processed_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump({
                'metadata': {
                    'dataset': 'Human3.6M',
                    'joints': self.h36m_joints,
                    'subjects': self.test_subjects,
                    'num_samples': len(test_data)
                },
                'data': test_data
            }, f)
        
        print(f"✅ Saved training data: {train_file}")
        print(f"✅ Saved test data: {test_file}")
        
        # Create a smaller sample for quick testing
        sample_size = min(1000, len(train_data))
        sample_data = train_data[:sample_size]
        
        sample_file = processed_dir / "sample_train.json"
        with open(sample_file, 'w') as f:
            json.dump({
                'metadata': {
                    'dataset': 'Human3.6M Sample',
                    'joints': self.h36m_joints,
                    'subjects': self.train_subjects,
                    'num_samples': len(sample_data),
                    'note': 'Small subset for quick testing'
                },
                'data': sample_data
            }, f)
        
        print(f"✅ Saved sample data ({sample_size} samples): {sample_file}")
        
        return processed_dir
    
    def create_dataset_config(self, processed_dir):
        """Create configuration file for the dataset"""
        config = {
            'dataset_name': 'Human3.6M',
            'root_dir': str(processed_dir),
            'train_file': 'train.json',
            'test_file': 'test.json',
            'sample_file': 'sample_train.json',
            'num_joints': 17,
            'joint_names': self.h36m_joints,
            'train_subjects': self.train_subjects,
            'test_subjects': self.test_subjects,
            'image_size': [1000, 1000],  # Original H36M image size
            'camera_params': {
                'note': 'Use actual camera calibration for each camera',
                'cameras': [1, 2, 3, 4]
            }
        }
        
        config_file = processed_dir / "dataset_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created dataset config: {config_file}")
        return config_file
    
    def process_dataset(self):
        """Main processing pipeline"""
        print("🚀 Processing Human3.6M Dataset")
        print("=" * 40)
        
        # Check if files exist
        if not self.check_files():
            self.print_download_instructions()
            return False
        
        # Extract files
        self.extract_files()
        
        # Load poses
        poses_2d, poses_3d = self.load_poses()
        if poses_2d is None or poses_3d is None:
            return False
        
        # Convert to training format
        train_data, test_data = self.convert_to_training_format(poses_2d, poses_3d)
        
        # Save processed data
        processed_dir = self.save_processed_data(train_data, test_data)
        
        # Create config
        self.create_dataset_config(processed_dir)
        
        print("\n🎉 Dataset processing complete!")
        print(f"📂 Processed data saved to: {processed_dir}")
        print(f"🔄 To use in training, update dataset_loader.py to load from:")
        print(f"   {processed_dir}/train.json")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Human3.6M Dataset Downloader and Processor')
    parser.add_argument('--download_dir', type=str, default='./data/human36m',
                       help='Directory to download/process dataset')
    parser.add_argument('--process', action='store_true',
                       help='Process already downloaded files')
    parser.add_argument('--instructions', action='store_true',
                       help='Show download instructions only')
    
    args = parser.parse_args()
    
    downloader = Human36MDownloader(args.download_dir)
    
    if args.instructions:
        downloader.print_download_instructions()
    elif args.process:
        success = downloader.process_dataset()
        if success:
            print("\n🔄 Next steps:")
            print("1. Update dataset_loader.py to use real Human3.6M data")
            print("2. Run training: python train.py --data_type human36m")
            print("3. Compare results with synthetic data baseline")
    else:
        downloader.print_download_instructions()

if __name__ == "__main__":
    main()