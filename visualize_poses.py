#!/usr/bin/env python3
"""
Comprehensive pose visualization with real dataset samples
Downloads sample images and demonstrates 3D pose estimation pipeline
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import urllib.request
import os
from pathlib import Path
import json
from typing import List, Tuple, Optional
import argparse

# Try to import mediapipe for real 2D pose detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe")

from model import Monocular3DPoseNet, CameraAwarePoseNet
from camera_utils import OV9281CameraCalibration, create_sample_camera_params
from inference import PoseInference

class PoseVisualizer:
    """Enhanced pose visualizer with real dataset support"""
    
    def __init__(self):
        self.sample_images_dir = Path('./sample_images')
        self.sample_images_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe if available
        self.mp_pose = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
        
        # Human3.6M to MediaPipe joint mapping (approximate)
        self.h36m_to_mp_mapping = {
            0: 23,   # Hip -> left_hip (approximate center)
            1: 24,   # RHip -> right_hip
            2: 26,   # RKnee -> right_knee
            3: 28,   # RFoot -> right_ankle
            4: 23,   # LHip -> left_hip
            5: 25,   # LKnee -> left_knee
            6: 27,   # LFoot -> left_ankle
            7: None, # Spine -> (interpolated)
            8: None, # Thorax -> (interpolated)
            9: None, # Neck -> (interpolated)
            10: 0,   # Head -> nose
            11: 11,  # LShoulder -> left_shoulder
            12: 13,  # LElbow -> left_elbow
            13: 15,  # LWrist -> left_wrist
            14: 12,  # RShoulder -> right_shoulder
            15: 14,  # RElbow -> right_elbow
            16: 16,  # RWrist -> right_wrist
        }
        
        # Colors for visualization
        self.colors = plt.cm.Set3(np.linspace(0, 1, 17))
        
        # Human3.6M skeleton connections
        self.skeleton = [
            (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # legs
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine to head
            (8, 11), (11, 12), (12, 13),  # left arm
            (8, 14), (14, 15), (15, 16)   # right arm
        ]
        
        self.joint_names = [
            'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
            'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 
            'LWrist', 'RShoulder', 'RElbow', 'RWrist'
        ]
    
    def download_sample_images(self):
        """Download sample pose images for testing"""
        print("🔹 Downloading sample pose images...")
        
        # Sample pose images (public domain / creative commons)
        sample_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Basketball_player_dribbling.jpg/640px-Basketball_player_dribbling.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Yoga_Poses.jpg/640px-Yoga_Poses.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Person_walking.jpg/480px-Person_walking.jpg"
        ]
        
        downloaded_files = []
        
        for i, url in enumerate(sample_urls):
            try:
                filename = f"sample_pose_{i+1}.jpg"
                filepath = self.sample_images_dir / filename
                
                if not filepath.exists():
                    print(f"  Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                
                downloaded_files.append(filepath)
                
            except Exception as e:
                print(f"  Failed to download {url}: {e}")
                # Create a synthetic image as fallback
                synthetic_img = self.create_synthetic_pose_image(i)
                filepath = self.sample_images_dir / f"synthetic_pose_{i+1}.jpg"
                cv2.imwrite(str(filepath), synthetic_img)
                downloaded_files.append(filepath)
        
        print(f"✅ Downloaded {len(downloaded_files)} sample images")
        return downloaded_files
    
    def create_synthetic_pose_image(self, idx: int) -> np.ndarray:
        """Create synthetic pose images as fallback"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Different poses based on index
        if idx == 0:  # Standing pose
            keypoints = [
                (320, 400), (310, 420), (300, 460), (295, 480),  # Right leg
                (330, 420), (340, 460), (345, 480),              # Left leg
                (320, 300), (320, 250), (320, 200), (320, 150),  # Spine to head
                (290, 260), (270, 290), (250, 320),              # Left arm
                (350, 260), (370, 290), (390, 320)               # Right arm
            ]
        elif idx == 1:  # Reaching pose
            keypoints = [
                (320, 400), (310, 420), (300, 460), (295, 480),  # Right leg
                (330, 420), (340, 460), (345, 480),              # Left leg
                (320, 300), (320, 250), (320, 200), (320, 150),  # Spine to head
                (290, 260), (260, 240), (230, 220),              # Left arm reaching up
                (350, 260), (380, 280), (410, 300)               # Right arm reaching
            ]
        else:  # Sitting pose
            keypoints = [
                (320, 350), (310, 350), (300, 380), (295, 400),  # Right leg
                (330, 350), (340, 380), (345, 400),              # Left leg
                (320, 320), (320, 280), (320, 240), (320, 200),  # Spine to head
                (290, 290), (270, 310), (250, 330),              # Left arm
                (350, 290), (370, 310), (390, 330)               # Right arm
            ]
        
        # Draw skeleton
        for i, (start_idx, end_idx) in enumerate(self.skeleton):
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_pt = keypoints[start_idx]
                end_pt = keypoints[end_idx]
                cv2.line(image, start_pt, end_pt, (0, 255, 0), 3)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            cv2.circle(image, kp, 5, (0, 0, 255), -1)
            cv2.putText(image, str(i), (kp[0]+5, kp[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(image, f"Synthetic Pose {idx+1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image
    
    def detect_2d_pose_mediapipe(self, image: np.ndarray) -> np.ndarray:
        """Detect 2D pose using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE or self.mp_pose is None:
            return self.detect_2d_pose_dummy(image)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.mp_pose.process(rgb_image)
        
        if results.pose_landmarks is None:
            print("  No pose detected by MediaPipe, using dummy pose")
            return self.detect_2d_pose_dummy(image)
        
        # Convert MediaPipe landmarks to Human3.6M format
        h, w = image.shape[:2]
        keypoints_2d = np.zeros((17, 2), dtype=np.float32)
        
        mp_landmarks = results.pose_landmarks.landmark
        
        for h36m_idx, mp_idx in self.h36m_to_mp_mapping.items():
            if mp_idx is not None and mp_idx < len(mp_landmarks):
                landmark = mp_landmarks[mp_idx]
                keypoints_2d[h36m_idx, 0] = landmark.x * w
                keypoints_2d[h36m_idx, 1] = landmark.y * h
            else:
                # Interpolate missing joints
                if h36m_idx == 7:  # Spine - between hip and thorax
                    keypoints_2d[7] = (keypoints_2d[0] + keypoints_2d[8]) / 2
                elif h36m_idx == 8:  # Thorax - between shoulders
                    left_shoulder = mp_landmarks[11]
                    right_shoulder = mp_landmarks[12]
                    keypoints_2d[8, 0] = (left_shoulder.x + right_shoulder.x) * w / 2
                    keypoints_2d[8, 1] = (left_shoulder.y + right_shoulder.y) * h / 2
                elif h36m_idx == 9:  # Neck - between thorax and head
                    keypoints_2d[9] = (keypoints_2d[8] + keypoints_2d[10]) / 2
        
        return keypoints_2d
    
    def detect_2d_pose_dummy(self, image: np.ndarray) -> np.ndarray:
        """Dummy 2D pose detection (fallback)"""
        h, w = image.shape[:2]
        
        # Create a reasonable dummy pose
        keypoints_2d = np.array([
            [w*0.5, h*0.65],    # Hip
            [w*0.45, h*0.7],    # RHip  
            [w*0.42, h*0.85],   # RKnee
            [w*0.40, h*0.95],   # RFoot
            [w*0.55, h*0.7],    # LHip
            [w*0.58, h*0.85],   # LKnee
            [w*0.60, h*0.95],   # LFoot
            [w*0.5, h*0.5],     # Spine
            [w*0.5, h*0.35],    # Thorax
            [w*0.5, h*0.25],    # Neck
            [w*0.5, h*0.15],    # Head
            [w*0.4, h*0.35],    # LShoulder
            [w*0.35, h*0.5],    # LElbow
            [w*0.30, h*0.6],    # LWrist
            [w*0.6, h*0.35],    # RShoulder
            [w*0.65, h*0.5],    # RElbow
            [w*0.70, h*0.6],    # RWrist
        ], dtype=np.float32)
        
        return keypoints_2d
    
    def visualize_2d_pose_advanced(self, image: np.ndarray, keypoints_2d: np.ndarray, 
                                  title: str = "2D Pose") -> np.ndarray:
        """Enhanced 2D pose visualization"""
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw skeleton with different colors for body parts
        limb_colors = {
            'spine': (255, 0, 0),      # Red
            'left_arm': (0, 255, 0),   # Green  
            'right_arm': (0, 0, 255),  # Blue
            'left_leg': (255, 255, 0), # Cyan
            'right_leg': (255, 0, 255) # Magenta
        }
        
        # Define limb groups
        limb_groups = {
            'spine': [(0, 7), (7, 8), (8, 9), (9, 10)],
            'left_arm': [(8, 11), (11, 12), (12, 13)],
            'right_arm': [(8, 14), (14, 15), (15, 16)],
            'left_leg': [(0, 4), (4, 5), (5, 6)],
            'right_leg': [(0, 1), (1, 2), (2, 3)]
        }
        
        # Draw limbs
        for limb_name, connections in limb_groups.items():
            color = limb_colors[limb_name]
            for start_idx, end_idx in connections:
                if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
                    start_pt = tuple(keypoints_2d[start_idx].astype(int))
                    end_pt = tuple(keypoints_2d[end_idx].astype(int))
                    cv2.line(vis_image, start_pt, end_pt, color, 3)
        
        # Draw keypoints with labels
        for i, (kp, joint_name) in enumerate(zip(keypoints_2d, self.joint_names)):
            pos = tuple(kp.astype(int))
            cv2.circle(vis_image, pos, 6, (255, 255, 255), -1)
            cv2.circle(vis_image, pos, 4, (0, 0, 0), -1)
            
            # Add joint labels
            cv2.putText(vis_image, f'{i}', (pos[0]+8, pos[1]-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(vis_image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_image
    
    def visualize_3d_pose_advanced(self, pose_3d: np.ndarray, 
                                  title: str = "3D Pose") -> plt.Figure:
        """Enhanced 3D pose visualization"""
        fig = plt.figure(figsize=(15, 5))
        
        # Multiple views of the 3D pose
        views = [
            {'elev': 15, 'azim': 70, 'title': 'Front View'},
            {'elev': 15, 'azim': 160, 'title': 'Side View'},
            {'elev': 70, 'azim': 70, 'title': 'Top View'}
        ]
        
        for i, view in enumerate(views):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            
            # Plot joints with different colors for body parts
            joint_colors = ['red', 'blue', 'blue', 'blue',  # Hip, right leg
                           'green', 'green', 'green',        # Left leg
                           'purple', 'purple', 'purple', 'purple',  # Spine
                           'orange', 'orange', 'orange',     # Left arm
                           'cyan', 'cyan', 'cyan']           # Right arm
            
            # Plot joints
            for j, (point, color, joint_name) in enumerate(zip(pose_3d, joint_colors, self.joint_names)):
                ax.scatter(point[0], point[1], point[2], c=color, s=80, alpha=0.8)
                ax.text(point[0], point[1], point[2], f'{j}', fontsize=8)
            
            # Plot skeleton
            for start_idx, end_idx in self.skeleton:
                if start_idx < len(pose_3d) and end_idx < len(pose_3d):
                    start_point = pose_3d[start_idx]
                    end_point = pose_3d[end_idx]
                    ax.plot([start_point[0], end_point[0]],
                           [start_point[1], end_point[1]],
                           [start_point[2], end_point[2]], 'b-', linewidth=2)
            
            # Set view and labels
            ax.view_init(elev=view['elev'], azim=view['azim'])
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(view['title'])
            
            # Set equal aspect ratio
            max_range = np.array([pose_3d[:,0].max()-pose_3d[:,0].min(),
                                 pose_3d[:,1].max()-pose_3d[:,1].min(),
                                 pose_3d[:,2].max()-pose_3d[:,2].min()]).max() / 2.0
            mid_x = (pose_3d[:,0].max()+pose_3d[:,0].min()) * 0.5
            mid_y = (pose_3d[:,1].max()+pose_3d[:,1].min()) * 0.5
            mid_z = (pose_3d[:,2].max()+pose_3d[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def process_and_visualize_image(self, image_path: Path, 
                                   pose_estimator: Optional[PoseInference] = None) -> dict:
        """Process single image and create comprehensive visualizations"""
        print(f"🔹 Processing {image_path.name}...")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ❌ Failed to load {image_path}")
            return None
        
        # Convert to grayscale for monochrome camera simulation
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect 2D pose
        keypoints_2d = self.detect_2d_pose_mediapipe(image)
        
        # Create visualizations
        vis_2d_original = self.visualize_2d_pose_advanced(image, keypoints_2d, 
                                                         f"2D Pose - {image_path.stem}")
        vis_2d_gray = self.visualize_2d_pose_advanced(gray_image, keypoints_2d, 
                                                     f"Monochrome - {image_path.stem}")
        
        result_data = {
            'image_path': image_path,
            'original_image': image,
            'gray_image': gray_image,
            'keypoints_2d': keypoints_2d,
            'vis_2d_original': vis_2d_original,
            'vis_2d_gray': vis_2d_gray,
            'pose_3d': None,
            'vis_3d': None
        }
        
        # If we have a trained model, predict 3D pose
        if pose_estimator is not None:
            try:
                _, pose_3d, _ = pose_estimator.process_image(gray_image)
                fig_3d = self.visualize_3d_pose_advanced(pose_3d, 
                                                        f"3D Pose - {image_path.stem}")
                
                result_data['pose_3d'] = pose_3d
                result_data['vis_3d'] = fig_3d
                
                print(f"  ✅ 3D pose predicted")
                print(f"    - 2D keypoints range: X[{keypoints_2d[:, 0].min():.1f}, {keypoints_2d[:, 0].max():.1f}], Y[{keypoints_2d[:, 1].min():.1f}, {keypoints_2d[:, 1].max():.1f}]")
                print(f"    - 3D pose range: X[{pose_3d[:, 0].min():.1f}, {pose_3d[:, 0].max():.1f}], Y[{pose_3d[:, 1].min():.1f}, {pose_3d[:, 1].max():.1f}], Z[{pose_3d[:, 2].min():.1f}, {pose_3d[:, 2].max():.1f}]")
                
            except Exception as e:
                print(f"  ⚠️ 3D pose prediction failed: {e}")
        
        return result_data
    
    def create_comparison_figure(self, results_list: List[dict]) -> plt.Figure:
        """Create comprehensive comparison figure"""
        n_images = len(results_list)
        
        # Determine if we have 3D predictions
        has_3d = any(r['pose_3d'] is not None for r in results_list)
        
        if has_3d:
            fig, axes = plt.subplots(3, n_images, figsize=(5*n_images, 15))
        else:
            fig, axes = plt.subplots(2, n_images, figsize=(5*n_images, 10))
        
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        
        for i, result in enumerate(results_list):
            if result is None:
                continue
            
            # Original image with 2D pose
            axes[0, i].imshow(cv2.cvtColor(result['vis_2d_original'], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f"2D Pose - {result['image_path'].stem}")
            axes[0, i].axis('off')
            
            # Grayscale with 2D pose
            axes[1, i].imshow(result['vis_2d_gray'], cmap='gray')
            axes[1, i].set_title(f"Monochrome Input - {result['image_path'].stem}")
            axes[1, i].axis('off')
            
            # 3D pose if available
            if has_3d and result['pose_3d'] is not None:
                # Create mini 3D plot
                ax_3d = fig.add_subplot(3, n_images, 2*n_images + i + 1, projection='3d')
                
                pose_3d = result['pose_3d']
                
                # Plot joints
                ax_3d.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], 
                             c='red', s=50, alpha=0.8)
                
                # Plot skeleton
                for start_idx, end_idx in self.skeleton:
                    if start_idx < len(pose_3d) and end_idx < len(pose_3d):
                        start_point = pose_3d[start_idx]
                        end_point = pose_3d[end_idx]
                        ax_3d.plot([start_point[0], end_point[0]],
                                  [start_point[1], end_point[1]],
                                  [start_point[2], end_point[2]], 'b-', linewidth=2)
                
                ax_3d.set_xlabel('X (mm)')
                ax_3d.set_ylabel('Y (mm)')
                ax_3d.set_zlabel('Z (mm)')
                ax_3d.set_title(f"3D Pose - {result['image_path'].stem}")
                ax_3d.view_init(elev=15, azim=70)
        
        plt.tight_layout()
        return fig

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Pose Visualization')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model for 3D prediction')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'camera_aware'])
    parser.add_argument('--download_samples', action='store_true',
                       help='Download sample images from internet')
    parser.add_argument('--custom_images', type=str, nargs='*',
                       help='Custom image paths to process')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Comprehensive Pose Visualization")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = PoseVisualizer()
    
    # Initialize pose estimator if model provided
    pose_estimator = None
    if args.model_path and Path(args.model_path).exists():
        print(f"🔹 Loading model from {args.model_path}...")
        try:
            pose_estimator = PoseInference(
                model_path=args.model_path,
                model_type=args.model_type
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    elif args.model_path:
        print(f"❌ Model not found at {args.model_path}")
    
    # Get images to process
    image_files = []
    
    if args.download_samples:
        image_files.extend(visualizer.download_sample_images())
    
    if args.custom_images:
        for img_path in args.custom_images:
            path = Path(img_path)
            if path.exists():
                image_files.append(path)
            else:
                print(f"❌ Image not found: {img_path}")
    
    # If no images specified, use existing samples or create synthetic ones
    if not image_files:
        print("🔹 No images specified, creating synthetic samples...")
        for i in range(3):
            synthetic_img = visualizer.create_synthetic_pose_image(i)
            filepath = visualizer.sample_images_dir / f"synthetic_pose_{i+1}.jpg"
            cv2.imwrite(str(filepath), synthetic_img)
            image_files.append(filepath)
    
    print(f"🔹 Processing {len(image_files)} images...")
    
    # Process all images
    results = []
    for img_path in image_files:
        result = visualizer.process_and_visualize_image(img_path, pose_estimator)
        if result:
            results.append(result)
            
            # Save individual results
            stem = img_path.stem
            cv2.imwrite(str(output_dir / f"{stem}_2d_original.jpg"), result['vis_2d_original'])
            cv2.imwrite(str(output_dir / f"{stem}_2d_gray.jpg"), result['vis_2d_gray'])
            
            if result['vis_3d']:
                result['vis_3d'].savefig(output_dir / f"{stem}_3d_pose.png", 
                                       dpi=150, bbox_inches='tight')
                plt.close(result['vis_3d'])
    
    # Create comparison figure
    if results:
        print("🔹 Creating comparison visualization...")
        comparison_fig = visualizer.create_comparison_figure(results)
        comparison_fig.savefig(output_dir / "pose_comparison.png", 
                             dpi=150, bbox_inches='tight')
        
        print(f"✅ Results saved to {output_dir}")
        print(f"  - Individual visualizations: {len(results)*2} files")
        print(f"  - Comparison figure: pose_comparison.png")
        
        # Show the comparison
        plt.show()
    
    # Print summary statistics
    if results:
        print("\n📊 Summary Statistics:")
        all_2d_x = np.concatenate([r['keypoints_2d'][:, 0] for r in results])
        all_2d_y = np.concatenate([r['keypoints_2d'][:, 1] for r in results])
        
        print(f"2D Keypoints:")
        print(f"  - X range: [{all_2d_x.min():.1f}, {all_2d_x.max():.1f}]")
        print(f"  - Y range: [{all_2d_y.min():.1f}, {all_2d_y.max():.1f}]")
        
        if any(r['pose_3d'] is not None for r in results):
            valid_3d = [r['pose_3d'] for r in results if r['pose_3d'] is not None]
            all_3d = np.concatenate(valid_3d)
            
            print(f"3D Poses ({len(valid_3d)} predictions):")
            print(f"  - X range: [{all_3d[:, 0].min():.1f}, {all_3d[:, 0].max():.1f}] mm")
            print(f"  - Y range: [{all_3d[:, 1].min():.1f}, {all_3d[:, 1].max():.1f}] mm")
            print(f"  - Z range: [{all_3d[:, 2].min():.1f}, {all_3d[:, 2].max():.1f}] mm")

if __name__ == "__main__":
    main()