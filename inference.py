import torch
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import json

from model import Monocular3DPoseNet, CameraAwarePoseNet
from camera_utils import OV9281CameraCalibration, create_sample_camera_params
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseInference:
    """Real-time 3D pose inference for OV9281 camera"""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'simple',
                 camera_matrix: np.ndarray = None,
                 dist_coeffs: np.ndarray = None,
                 device: torch.device = None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.eval()
        
        # Camera calibration
        if camera_matrix is not None and dist_coeffs is not None:
            self.camera_calibration = OV9281CameraCalibration(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
        else:
            # Use default parameters
            camera_matrix, dist_coeffs = create_sample_camera_params()
            self.camera_calibration = OV9281CameraCalibration(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
        
        # Human3.6M skeleton connections for visualization
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
    
    def _load_model(self, model_path: str, model_type: str):
        """Load trained model"""
        if model_type == 'simple':
            model = Monocular3DPoseNet()
        elif model_type == 'camera_aware':
            model = CameraAwarePoseNet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Loaded {model_type} model from {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        return model
    
    def detect_2d_pose(self, image: np.ndarray) -> np.ndarray:
        """
        Detect 2D pose keypoints from image
        For now, returns dummy keypoints - integrate with your 2D pose detector
        """
        h, w = image.shape[:2]
        
        # Dummy 2D keypoints - replace with actual 2D pose detection
        # You can integrate MediaPipe, OpenPose, or other 2D pose detectors here
        keypoints_2d = np.array([
            [w*0.5, h*0.6],    # Hip
            [w*0.45, h*0.7],   # RHip  
            [w*0.42, h*0.85],  # RKnee
            [w*0.40, h*0.95],  # RFoot
            [w*0.55, h*0.7],   # LHip
            [w*0.58, h*0.85],  # LKnee
            [w*0.60, h*0.95],  # LFoot
            [w*0.5, h*0.45],   # Spine
            [w*0.5, h*0.35],   # Thorax
            [w*0.5, h*0.25],   # Neck
            [w*0.5, h*0.15],   # Head
            [w*0.4, h*0.3],    # LShoulder
            [w*0.35, h*0.45],  # LElbow
            [w*0.30, h*0.55],  # LWrist
            [w*0.6, h*0.3],    # RShoulder
            [w*0.65, h*0.45],  # RElbow
            [w*0.70, h*0.55],  # RWrist
        ], dtype=np.float32)
        
        return keypoints_2d
    
    def predict_3d_pose(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """Predict 3D pose from 2D keypoints"""
        
        # Normalize 2D keypoints
        normalized_kp = self.camera_calibration.normalize_2d_keypoints(keypoints_2d)
        
        # Convert to tensor
        kp_tensor = torch.from_numpy(normalized_kp).float().unsqueeze(0).to(self.device)
        
        # Get camera parameters
        camera_params = self.camera_calibration.get_camera_params_tensor()
        camera_params = camera_params.unsqueeze(0).to(self.device)
        
        # Predict 3D pose
        with torch.no_grad():
            if hasattr(self.model, 'lifting_net'):  # CameraAwarePoseNet
                pred_3d = self.model(kp_tensor, camera_params)
            else:  # Monocular3DPoseNet
                pred_3d = self.model(kp_tensor, camera_params)
        
        # Convert back to numpy
        pose_3d = pred_3d.cpu().numpy().squeeze()
        
        return pose_3d
    
    def visualize_pose_3d(self, pose_3d: np.ndarray, title: str = "3D Pose") -> plt.Figure:
        """Visualize 3D pose"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot joints
        ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # Plot skeleton
        for start_idx, end_idx in self.skeleton:
            if start_idx < len(pose_3d) and end_idx < len(pose_3d):
                start_point = pose_3d[start_idx]
                end_point = pose_3d[end_idx]
                ax.plot([start_point[0], end_point[0]],
                       [start_point[1], end_point[1]],
                       [start_point[2], end_point[2]], 'b-', linewidth=2)
        
        # Add joint labels
        for i, (joint_name, point) in enumerate(zip(self.joint_names, pose_3d)):
            ax.text(point[0], point[1], point[2], f'{i}', fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        
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
        
        return fig
    
    def visualize_pose_2d(self, image: np.ndarray, keypoints_2d: np.ndarray) -> np.ndarray:
        """Visualize 2D pose on image"""
        vis_image = image.copy()
        if len(vis_image.shape) == 2:  # Grayscale
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw skeleton
        for start_idx, end_idx in self.skeleton:
            if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
                start_point = tuple(keypoints_2d[start_idx].astype(int))
                end_point = tuple(keypoints_2d[end_idx].astype(int))
                cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints_2d):
            cv2.circle(vis_image, tuple(kp.astype(int)), 4, (0, 0, 255), -1)
            cv2.putText(vis_image, str(i), tuple(kp.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return vis_image
    
    def process_image(self, image: np.ndarray) -> tuple:
        """Process single image and return 2D and 3D poses"""
        
        # Undistort image
        undistorted = self.camera_calibration.undistort_image(image)
        
        # Detect 2D pose
        keypoints_2d = self.detect_2d_pose(undistorted)
        
        # Predict 3D pose
        pose_3d = self.predict_3d_pose(keypoints_2d)
        
        return keypoints_2d, pose_3d, undistorted
    
    def process_video(self, video_path: str, output_path: str = None):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for monochrome camera simulation
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process frame
            start_time = time.time()
            keypoints_2d, pose_3d, undistorted = self.process_image(gray_frame)
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            frame_count += 1
            
            # Visualize 2D pose
            vis_frame = self.visualize_pose_2d(undistorted, keypoints_2d)
            
            # Add FPS info
            fps_text = f"FPS: {1.0/processing_time:.1f}"
            cv2.putText(vis_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('3D Pose Estimation', vis_frame)
            
            # Write to output video
            if output_path:
                out.write(vis_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Processed {frame_count} frames")
        print(f"Average processing speed: {avg_fps:.2f} FPS")
    
    def process_camera(self, camera_id: int = 0):
        """Process live camera feed"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Set camera properties for OV9281 if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Press 'q' to quit, 's' to save current pose")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process frame
            start_time = time.time()
            keypoints_2d, pose_3d, undistorted = self.process_image(gray_frame)
            end_time = time.time()
            
            # Visualize
            vis_frame = self.visualize_pose_2d(undistorted, keypoints_2d)
            
            # Add FPS
            fps = 1.0 / (end_time - start_time)
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Live 3D Pose Estimation', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current 3D pose
                timestamp = int(time.time())
                pose_data = {
                    'timestamp': timestamp,
                    'keypoints_2d': keypoints_2d.tolist(),
                    'pose_3d': pose_3d.tolist(),
                    'joint_names': self.joint_names
                }
                
                filename = f'pose_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump(pose_data, f, indent=2)
                
                print(f"Pose saved to {filename}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='3D Pose Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'camera_aware'],
                       help='Type of model to load')
    parser.add_argument('--input', type=str, default='camera',
                       help='Input source: "camera", video file path, or image file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (for video input only)')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera ID for live inference')
    parser.add_argument('--camera_config', type=str, default=None,
                       help='Path to camera calibration JSON file')
    
    args = parser.parse_args()
    
    # Load camera calibration if provided
    camera_matrix = None
    dist_coeffs = None
    
    if args.camera_config:
        calibration = OV9281CameraCalibration()
        calibration.load_calibration(args.camera_config)
        camera_matrix = calibration.camera_matrix
        dist_coeffs = calibration.dist_coeffs
    
    # Create inference object
    pose_estimator = PoseInference(
        model_path=args.model_path,
        model_type=args.model_type,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )
    
    # Run inference
    if args.input == 'camera':
        pose_estimator.process_camera(args.camera_id)
    elif args.input.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        pose_estimator.process_video(args.input, args.output)
    elif args.input.endswith(('.jpg', '.png', '.jpeg')):
        # Process single image
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Cannot load image: {args.input}")
            return
        
        keypoints_2d, pose_3d, undistorted = pose_estimator.process_image(image)
        
        # Visualize results
        vis_2d = pose_estimator.visualize_pose_2d(undistorted, keypoints_2d)
        fig_3d = pose_estimator.visualize_pose_3d(pose_3d)
        
        # Save results
        cv2.imwrite('pose_2d_result.jpg', vis_2d)
        fig_3d.savefig('pose_3d_result.png', dpi=150)
        
        print("Results saved: pose_2d_result.jpg, pose_3d_result.png")
        
        # Show results
        cv2.imshow('2D Pose', vis_2d)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()