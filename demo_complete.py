#!/usr/bin/env python3
"""
Complete demonstration of the 3D Pose Estimation System
Shows the entire pipeline from 2D detection to 3D pose estimation
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from model import Monocular3DPoseNet, CameraAwarePoseNet
from camera_utils import OV9281CameraCalibration, create_sample_camera_params
from inference import PoseInference
from visualize_poses import PoseVisualizer

def create_demo_image_with_pose() -> np.ndarray:
    """Create a realistic demo image with a person-like pose"""
    
    # Create a 720p grayscale image (simulating OV9281 output)
    image = np.zeros((720, 1280), dtype=np.uint8)
    
    # Add some background texture
    noise = np.random.normal(50, 20, image.shape).astype(np.uint8)
    image = np.clip(image + noise, 0, 255)
    
    # Define a realistic standing pose (pixel coordinates)
    # These represent Human3.6M joint order
    pose_keypoints = np.array([
        [640, 500],    # 0: Hip (center)
        [610, 520],    # 1: RHip
        [590, 580],    # 2: RKnee  
        [585, 640],    # 3: RFoot
        [670, 520],    # 4: LHip
        [690, 580],    # 5: LKnee
        [695, 640],    # 6: LFoot
        [640, 450],    # 7: Spine
        [640, 380],    # 8: Thorax
        [640, 320],    # 9: Neck
        [640, 280],    # 10: Head
        [600, 360],    # 11: LShoulder
        [570, 420],    # 12: LElbow
        [550, 480],    # 13: LWrist
        [680, 360],    # 14: RShoulder
        [710, 420],    # 15: RElbow
        [730, 480],    # 16: RWrist
    ], dtype=np.float32)
    
    # Human3.6M skeleton connections
    skeleton = [
        (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # legs
        (0, 7), (7, 8), (8, 9), (9, 10),  # spine to head
        (8, 11), (11, 12), (12, 13),  # left arm
        (8, 14), (14, 15), (15, 16)   # right arm
    ]
    
    # Draw skeleton with varying thickness for depth perception
    for i, (start_idx, end_idx) in enumerate(skeleton):
        start_pt = tuple(pose_keypoints[start_idx].astype(int))
        end_pt = tuple(pose_keypoints[end_idx].astype(int))
        
        # Vary line thickness and brightness for more realistic appearance
        if i < 6:  # legs
            thickness = 4
            brightness = 200
        elif i < 10:  # spine
            thickness = 5
            brightness = 220
        else:  # arms
            thickness = 3
            brightness = 180
            
        cv2.line(image, start_pt, end_pt, brightness, thickness)
    
    # Draw joints as circles
    for i, kp in enumerate(pose_keypoints):
        pos = tuple(kp.astype(int))
        
        # Different sizes for different joint types
        if i == 0:  # Hip - larger
            radius = 8
        elif i == 10:  # Head - larger
            radius = 12
        elif i in [8, 9]:  # Thorax, neck - medium
            radius = 6
        else:  # Other joints - smaller
            radius = 5
            
        cv2.circle(image, pos, radius, 255, -1)  # White filled circle
        cv2.circle(image, pos, radius+1, 100, 1)  # Dark outline
    
    # Add some random noise to simulate camera noise
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply slight blur to simulate camera optics
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    return image

def demonstrate_pipeline():
    """Demonstrate the complete 3D pose estimation pipeline"""
    
    print("🚀 Complete 3D Pose Estimation Pipeline Demonstration")
    print("=" * 60)
    
    # Check if we have a trained model
    model_path = Path('./checkpoints/best_model.pth')
    if not model_path.exists():
        print("❌ No trained model found!")
        print("   Please run: python train.py --epochs 10")
        print("   This will create a quick demo model.")
        return
    
    # 1. Create camera calibration
    print("🔹 Step 1: Setting up camera calibration for OV9281...")
    camera_matrix, dist_coeffs = create_sample_camera_params()
    calibration = OV9281CameraCalibration(
        image_width=1280,
        image_height=720,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )
    
    print(f"   Camera parameters:")
    print(f"   - Focal lengths: fx={calibration.fx:.1f}, fy={calibration.fy:.1f}")
    print(f"   - Principal point: cx={calibration.cx:.1f}, cy={calibration.cy:.1f}")
    print(f"   - FOV: ~160 degrees (wide angle)")
    
    # 2. Load trained model
    print("\n🔹 Step 2: Loading trained 3D pose estimation model...")
    pose_estimator = PoseInference(
        model_path=str(model_path),
        model_type='simple'
    )
    print(f"   ✅ Model loaded successfully")
    
    # 3. Create demo image
    print("\n🔹 Step 3: Creating demo image (simulating OV9281 output)...")
    demo_image = create_demo_image_with_pose()
    demo_path = Path('./demo_pose_image.jpg')
    cv2.imwrite(str(demo_path), demo_image)
    print(f"   ✅ Demo image created: {demo_path}")
    
    # 4. Process through complete pipeline
    print("\n🔹 Step 4: Processing through complete pipeline...")
    
    # Step 4a: 2D pose detection (dummy for now)
    print("   4a: Detecting 2D keypoints...")
    visualizer = PoseVisualizer()
    keypoints_2d = visualizer.detect_2d_pose_dummy(demo_image)
    print(f"       ✅ Detected {len(keypoints_2d)} keypoints")
    
    # Step 4b: Camera undistortion
    print("   4b: Applying camera undistortion...")
    undistorted_image = calibration.undistort_image(demo_image)
    print("       ✅ Image undistorted")
    
    # Step 4c: 3D pose prediction
    print("   4c: Predicting 3D pose from 2D keypoints...")
    _, pose_3d, _ = pose_estimator.process_image(demo_image)
    print(f"       ✅ 3D pose predicted")
    print(f"       - Range: X[{pose_3d[:, 0].min():.1f}, {pose_3d[:, 0].max():.1f}] mm")
    print(f"                Y[{pose_3d[:, 1].min():.1f}, {pose_3d[:, 1].max():.1f}] mm") 
    print(f"                Z[{pose_3d[:, 2].min():.1f}, {pose_3d[:, 2].max():.1f}] mm")
    
    # 5. Create comprehensive visualizations
    print("\n🔹 Step 5: Creating comprehensive visualizations...")
    
    # 2D visualization
    vis_2d = visualizer.visualize_2d_pose_advanced(demo_image, keypoints_2d, 
                                                  "2D Pose Detection")
    
    # 3D visualization  
    fig_3d = visualizer.visualize_3d_pose_advanced(pose_3d, "3D Pose Estimation")
    
    # Save results
    results_dir = Path('./demo_results')
    results_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(results_dir / 'original_image.jpg'), demo_image)
    cv2.imwrite(str(results_dir / 'undistorted_image.jpg'), undistorted_image)
    cv2.imwrite(str(results_dir / '2d_pose_detection.jpg'), vis_2d)
    fig_3d.savefig(results_dir / '3d_pose_estimation.png', dpi=150, bbox_inches='tight')
    
    print(f"   ✅ Visualizations saved to: {results_dir}")
    
    # 6. Performance analysis
    print("\n🔹 Step 6: Performance Analysis...")
    
    # Measure inference time
    import time
    times = []
    for i in range(10):
        start = time.time()
        pose_estimator.process_image(demo_image)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"   Average inference time: {avg_time*1000:.1f} ms")
    print(f"   Theoretical max FPS: {fps:.1f}")
    print(f"   Model size: ~{sum(p.numel() for p in pose_estimator.model.parameters())/1e6:.1f}M parameters")
    
    # 7. Show results
    print("\n🔹 Step 7: Displaying results...")
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(demo_image, cmap='gray')
    axes[0, 0].set_title('Original Image (OV9281 Simulation)')
    axes[0, 0].axis('off')
    
    # 2D pose detection
    axes[0, 1].imshow(vis_2d, cmap='gray' if len(vis_2d.shape) == 2 else None)
    axes[0, 1].set_title('2D Pose Detection')
    axes[0, 1].axis('off')
    
    # 3D pose visualization (simplified)
    ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax_3d.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='red', s=50)
    
    # Draw skeleton in 3D
    skeleton = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (9, 10),
                (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16)]
    
    for start_idx, end_idx in skeleton:
        if start_idx < len(pose_3d) and end_idx < len(pose_3d):
            start_point = pose_3d[start_idx]
            end_point = pose_3d[end_idx]
            ax_3d.plot([start_point[0], end_point[0]],
                      [start_point[1], end_point[1]],
                      [start_point[2], end_point[2]], 'b-', linewidth=2)
    
    ax_3d.set_xlabel('X (mm)')
    ax_3d.set_ylabel('Y (mm)')
    ax_3d.set_zlabel('Z (mm)')
    ax_3d.set_title('3D Pose Estimation')
    ax_3d.view_init(elev=15, azim=70)
    
    # Performance info
    axes[1, 1].text(0.1, 0.8, f"Performance Metrics:\n\n" + 
                    f"• Average inference: {avg_time*1000:.1f} ms\n" +
                    f"• Theoretical FPS: {fps:.1f}\n" +
                    f"• Model parameters: {sum(p.numel() for p in pose_estimator.model.parameters())/1e6:.1f}M\n" +
                    f"• Input resolution: 1280x720\n" +
                    f"• Camera FOV: ~160°\n" +
                    f"• 3D joints: 17\n\n" +
                    f"Next Steps:\n" +
                    f"• Integrate real 2D detector\n" +
                    f"• Calibrate actual camera\n" +
                    f"• Train on real data\n" +
                    f"• Optimize for real-time",
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.suptitle('3D Pose Estimation System - Complete Pipeline', fontsize=16)
    plt.tight_layout()
    plt.savefig(results_dir / 'complete_pipeline_demo.png', dpi=150, bbox_inches='tight')
    
    print("   ✅ Complete pipeline visualization created")
    
    # Show the plot
    plt.show()
    
    print("\n🎉 Demonstration Complete!")
    print(f"   All results saved to: {results_dir}")
    print(f"   Key files:")
    print(f"   - complete_pipeline_demo.png: Full pipeline overview")
    print(f"   - 3d_pose_estimation.png: Detailed 3D visualization")
    print(f"   - 2d_pose_detection.jpg: 2D keypoint detection")
    
    print(f"\n🔄 To use with real OV9281 camera:")
    print(f"   1. Replace dummy 2D detector with MediaPipe/OpenPose")
    print(f"   2. Calibrate your camera: python -c \"from camera_utils import *; calibrate_camera_from_chessboard('./calib_images/')\"")
    print(f"   3. Train on real data with actual poses")
    print(f"   4. Run real-time inference: python inference.py --model_path ./checkpoints/best_model.pth --input camera")

def main():
    parser = argparse.ArgumentParser(description='Complete Pipeline Demonstration')
    parser.add_argument('--interactive', action='store_true',
                       help='Show interactive plots')
    
    args = parser.parse_args()
    
    try:
        demonstrate_pipeline()
    except KeyboardInterrupt:
        print("\n⏸️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("   Make sure you have run training first: python train.py --epochs 10")

if __name__ == "__main__":
    main()