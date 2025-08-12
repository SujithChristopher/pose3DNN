#!/usr/bin/env python3
"""
Quick start script for 3D pose estimation with OV9281 camera
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from model import Monocular3DPoseNet, CameraAwarePoseNet
from camera_utils import OV9281CameraCalibration, create_sample_camera_params
from dataset_loader import Human36MDataset
import argparse

def demo_dataset():
    """Demonstrate the dataset creation and loading"""
    print("🔹 Creating synthetic dataset...")
    dataset = Human36MDataset(download=True)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Show a sample
    sample = dataset[0]
    print(f"Sample data structure:")
    print(f"  - Image shape: {sample['image'].shape}")
    print(f"  - 2D keypoints shape: {sample['keypoints_2d'].shape}")
    print(f"  - 3D keypoints shape: {sample['keypoints_3d'].shape}")
    print(f"  - Subject: {sample['subject']}")
    print(f"  - Action: {sample['action']}")

def demo_camera_calibration():
    """Demonstrate camera calibration utilities"""
    print("🔹 Setting up OV9281 camera calibration...")
    
    # Create sample camera parameters for 160-degree FOV
    camera_matrix, dist_coeffs = create_sample_camera_params()
    
    # Initialize calibration object
    calibration = OV9281CameraCalibration(
        image_width=1280, 
        image_height=720,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )
    
    print(f"✅ Camera calibration initialized")
    print(f"  - Focal lengths: fx={calibration.fx:.1f}, fy={calibration.fy:.1f}")
    print(f"  - Principal point: cx={calibration.cx:.1f}, cy={calibration.cy:.1f}")
    print(f"  - Distortion coeffs: {calibration.dist_coeffs}")
    
    # Save calibration for later use
    calibration.save_calibration('./camera_calibration.json')
    print("✅ Camera calibration saved to camera_calibration.json")
    
    return calibration

def demo_model():
    """Demonstrate model creation and inference"""
    print("🔹 Creating 3D pose estimation models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    simple_model = Monocular3DPoseNet().to(device)
    camera_aware_model = CameraAwarePoseNet().to(device)
    
    print(f"✅ Simple model: {sum(p.numel() for p in simple_model.parameters()):,} parameters")
    print(f"✅ Camera-aware model: {sum(p.numel() for p in camera_aware_model.parameters()):,} parameters")
    
    # Test forward pass with dummy data
    batch_size = 2
    num_joints = 17
    
    keypoints_2d = torch.randn(batch_size, num_joints, 2).to(device)
    camera_params = torch.randn(batch_size, 9).to(device)
    
    with torch.no_grad():
        pred_3d_simple = simple_model(keypoints_2d, camera_params)
        pred_3d_camera = camera_aware_model(keypoints_2d, camera_params)
    
    print(f"✅ Forward pass successful")
    print(f"  - Simple model output: {pred_3d_simple.shape}")
    print(f"  - Camera-aware output: {pred_3d_camera.shape}")
    
    return simple_model, camera_aware_model, device

def quick_train_demo(model, device, epochs=5):
    """Quick training demonstration"""
    print(f"🔹 Running quick training demo ({epochs} epochs)...")
    
    # Import here to avoid circular imports
    from dataset_loader import create_data_loaders
    from train import PoseTrainer
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(batch_size=4, num_workers=0)
    
    # Create trainer
    trainer = PoseTrainer(model, device, learning_rate=1e-3)
    
    # Train for a few epochs
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=epochs, save_dir='./demo_checkpoints')
    
    print("✅ Quick training completed!")
    return trainer

def demo_inference():
    """Demonstrate inference on a synthetic image"""
    print("🔹 Demonstrating inference...")
    
    # Check if we have a trained model
    model_path = Path('./demo_checkpoints/best_model.pth')
    if not model_path.exists():
        print("❌ No trained model found. Run training first.")
        return
    
    from inference import PoseInference
    
    # Create inference object
    pose_estimator = PoseInference(
        model_path=str(model_path),
        model_type='simple'
    )
    
    # Create a synthetic test image
    test_image = np.zeros((720, 1280), dtype=np.uint8)
    
    # Add some synthetic pose-like features
    cv2.circle(test_image, (640, 400), 5, 255, -1)  # Head
    cv2.circle(test_image, (640, 450), 5, 255, -1)  # Neck
    cv2.circle(test_image, (640, 500), 5, 255, -1)  # Torso
    cv2.line(test_image, (640, 450), (600, 480), 255, 2)  # Left arm
    cv2.line(test_image, (640, 450), (680, 480), 255, 2)  # Right arm
    cv2.line(test_image, (640, 500), (620, 600), 255, 2)  # Left leg
    cv2.line(test_image, (640, 500), (660, 600), 255, 2)  # Right leg
    
    # Process image
    keypoints_2d, pose_3d, undistorted = pose_estimator.process_image(test_image)
    
    # Visualize results
    vis_2d = pose_estimator.visualize_pose_2d(undistorted, keypoints_2d)
    fig_3d = pose_estimator.visualize_pose_3d(pose_3d, "Predicted 3D Pose")
    
    # Save results
    cv2.imwrite('demo_pose_2d.jpg', vis_2d)
    fig_3d.savefig('demo_pose_3d.png', dpi=150, bbox_inches='tight')
    
    print("✅ Inference completed!")
    print("  - 2D pose saved: demo_pose_2d.jpg")
    print("  - 3D pose saved: demo_pose_3d.png")
    
    # Show some statistics
    print(f"  - 2D keypoints range: X[{keypoints_2d[:, 0].min():.1f}, {keypoints_2d[:, 0].max():.1f}], Y[{keypoints_2d[:, 1].min():.1f}, {keypoints_2d[:, 1].max():.1f}]")
    print(f"  - 3D pose range: X[{pose_3d[:, 0].min():.1f}, {pose_3d[:, 0].max():.1f}], Y[{pose_3d[:, 1].min():.1f}, {pose_3d[:, 1].max():.1f}], Z[{pose_3d[:, 2].min():.1f}, {pose_3d[:, 2].max():.1f}]")

def main():
    parser = argparse.ArgumentParser(description='Quick start demo for 3D pose estimation')
    parser.add_argument('--demo', type=str, default='all', 
                       choices=['all', 'dataset', 'camera', 'model', 'train', 'inference'],
                       help='Which demo to run')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs for training demo')
    
    args = parser.parse_args()
    
    print("🚀 3D Pose Estimation Quick Start Demo")
    print("=" * 50)
    
    if args.demo in ['all', 'dataset']:
        demo_dataset()
        print()
    
    if args.demo in ['all', 'camera']:
        calibration = demo_camera_calibration()
        print()
    
    if args.demo in ['all', 'model']:
        simple_model, camera_aware_model, device = demo_model()
        print()
    
    if args.demo in ['all', 'train']:
        if 'simple_model' not in locals():
            simple_model, _, device = demo_model()
        trainer = quick_train_demo(simple_model, device, epochs=args.epochs)
        print()
    
    if args.demo in ['all', 'inference']:
        demo_inference()
        print()
    
    print("✅ Demo completed!")
    print("\nNext steps:")
    print("1. Replace dummy 2D keypoint detection with real detector (MediaPipe/OpenPose)")
    print("2. Calibrate your actual OV9281 camera")
    print("3. Collect and train on real pose data")
    print("4. Fine-tune for your specific use case")

if __name__ == "__main__":
    main()