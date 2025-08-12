import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
import json

class OV9281CameraCalibration:
    """
    Camera calibration utilities for OV9281 160-degree monochrome camera
    """
    
    def __init__(self, 
                 image_width: int = 1280, 
                 image_height: int = 720,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        
        self.image_width = image_width
        self.image_height = image_height
        
        # Use provided camera parameters or defaults
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            # Default camera matrix for 160-degree FOV (you should calibrate this)
            # For 160-degree FOV, focal length is typically much smaller
            focal_length = min(image_width, image_height) * 0.3  # Adjusted for wide FOV
            self.camera_matrix = np.array([
                [focal_length, 0, image_width/2],
                [0, focal_length, image_height/2],
                [0, 0, 1]
            ], dtype=np.float32)
        
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
        else:
            # Default distortion coefficients (you should calibrate these)
            # Wide-angle cameras typically have significant barrel distortion
            self.dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.05], dtype=np.float32)
        
        # Store for easy access
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
    
    def set_camera_parameters(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set calibrated camera parameters"""
        self.camera_matrix = camera_matrix.astype(np.float32)
        self.dist_coeffs = dist_coeffs.astype(np.float32)
        
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        print(f"Updated camera parameters:")
        print(f"Camera matrix:\n{self.camera_matrix}")
        print(f"Distortion coefficients: {self.dist_coeffs}")
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using camera calibration"""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort 2D points using camera calibration
        
        Args:
            points: (N, 2) array of distorted image points
        
        Returns:
            undistorted_points: (N, 2) array of undistorted points
        """
        points = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, 
                                        P=self.camera_matrix)
        return undistorted.reshape(-1, 2)
    
    def normalize_2d_keypoints(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Normalize 2D keypoints to [-1, 1] range for neural network input
        
        Args:
            keypoints_2d: (N, 2) array of 2D keypoints in pixel coordinates
        
        Returns:
            normalized_keypoints: (N, 2) array of normalized keypoints
        """
        normalized = keypoints_2d.copy()
        
        # Normalize to [0, 1]
        normalized[:, 0] = keypoints_2d[:, 0] / self.image_width
        normalized[:, 1] = keypoints_2d[:, 1] / self.image_height
        
        # Normalize to [-1, 1]
        normalized = 2.0 * normalized - 1.0
        
        return normalized
    
    def denormalize_2d_keypoints(self, normalized_keypoints: np.ndarray) -> np.ndarray:
        """Convert normalized keypoints back to pixel coordinates"""
        # Convert from [-1, 1] to [0, 1]
        denormalized = (normalized_keypoints + 1.0) / 2.0
        
        # Convert to pixel coordinates
        denormalized[:, 0] *= self.image_width
        denormalized[:, 1] *= self.image_height
        
        return denormalized
    
    def get_camera_params_tensor(self) -> torch.Tensor:
        """Get camera parameters as PyTorch tensor for neural network"""
        params = torch.tensor([
            self.fx, self.fy, self.cx, self.cy,
            self.dist_coeffs[0], self.dist_coeffs[1], self.dist_coeffs[2],
            self.dist_coeffs[3], self.dist_coeffs[4]
        ], dtype=torch.float32)
        
        return params
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: (N, 3) array of 3D points in camera coordinate system
        
        Returns:
            points_2d: (N, 2) array of 2D image coordinates
        """
        # Use OpenCV projection
        rvec = np.zeros(3, dtype=np.float32)  # No rotation (already in camera coords)
        tvec = np.zeros(3, dtype=np.float32)  # No translation
        
        points_2d, _ = cv2.projectPoints(points_3d.reshape(-1, 1, 3), 
                                       rvec, tvec, 
                                       self.camera_matrix, 
                                       self.dist_coeffs)
        
        return points_2d.reshape(-1, 2)
    
    def save_calibration(self, filepath: str):
        """Save camera calibration to file"""
        calibration_data = {
            'image_width': self.image_width,
            'image_height': self.image_height,
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Camera calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load camera calibration from file"""
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)
        
        self.image_width = calibration_data['image_width']
        self.image_height = calibration_data['image_height']
        self.camera_matrix = np.array(calibration_data['camera_matrix'], dtype=np.float32)
        self.dist_coeffs = np.array(calibration_data['dist_coeffs'], dtype=np.float32)
        
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        print(f"Camera calibration loaded from {filepath}")

def calibrate_camera_from_chessboard(images_path: str, 
                                   chessboard_size: Tuple[int, int] = (9, 6),
                                   square_size: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate camera using chessboard images
    
    Args:
        images_path: Path to directory containing chessboard images
        chessboard_size: (width, height) of internal chessboard corners
        square_size: Size of chessboard squares in mm
    
    Returns:
        camera_matrix, dist_coeffs
    """
    import glob
    import os
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    image_files = glob.glob(os.path.join(images_path, '*.jpg')) + \
                  glob.glob(os.path.join(images_path, '*.png'))
    
    if not image_files:
        raise ValueError(f"No calibration images found in {images_path}")
    
    print(f"Found {len(image_files)} calibration images")
    
    successful_detections = 0
    
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            successful_detections += 1
            objpoints.append(objp)
            
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                                              30, 0.001))
            imgpoints.append(corners2)
            
            print(f"✓ Detected corners in {os.path.basename(image_file)}")
        else:
            print(f"✗ Failed to detect corners in {os.path.basename(image_file)}")
    
    if successful_detections < 5:
        raise ValueError(f"Need at least 5 successful detections, got {successful_detections}")
    
    print(f"Successfully detected corners in {successful_detections} images")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if not ret:
        raise RuntimeError("Camera calibration failed")
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                        camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {mean_error:.3f} pixels")
    
    return camera_matrix, dist_coeffs

def create_sample_camera_params():
    """Create sample camera parameters for OV9281"""
    # Sample parameters for OV9281 with 160-degree FOV
    # These are estimates - you should calibrate your actual camera
    
    image_width, image_height = 1280, 720
    
    # For 160-degree FOV, focal length is much smaller
    focal_length = min(image_width, image_height) * 0.25  # Very wide FOV
    
    camera_matrix = np.array([
        [focal_length, 0, image_width/2],
        [0, focal_length, image_height/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Wide-angle cameras typically have significant barrel distortion
    dist_coeffs = np.array([0.15, -0.3, 0.0, 0.0, 0.1], dtype=np.float32)
    
    return camera_matrix, dist_coeffs

if __name__ == "__main__":
    # Test camera calibration utilities
    camera_matrix, dist_coeffs = create_sample_camera_params()
    
    # Create calibration object
    calibration = OV9281CameraCalibration(
        camera_matrix=camera_matrix, 
        dist_coeffs=dist_coeffs
    )
    
    print("Camera calibration created successfully")
    print(f"Focal lengths: fx={calibration.fx:.2f}, fy={calibration.fy:.2f}")
    print(f"Principal point: cx={calibration.cx:.2f}, cy={calibration.cy:.2f}")
    print(f"Distortion coefficients: {calibration.dist_coeffs}")
    
    # Test parameter tensor creation
    param_tensor = calibration.get_camera_params_tensor()
    print(f"Camera parameter tensor shape: {param_tensor.shape}")
    print(f"Camera parameter tensor: {param_tensor}")
    
    # Test point normalization
    test_points = np.array([[640, 360], [100, 200], [1180, 620]])
    normalized = calibration.normalize_2d_keypoints(test_points)
    denormalized = calibration.denormalize_2d_keypoints(normalized)
    
    print(f"Original points: {test_points}")
    print(f"Normalized: {normalized}")
    print(f"Denormalized: {denormalized}")
    print(f"Roundtrip error: {np.mean(np.abs(test_points - denormalized)):.6f}")