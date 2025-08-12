import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

class ResidualBlock(nn.Module):
    """Residual block for the lifting network"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.25):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        
        return self.relu(out + residual)

class Monocular3DPoseNet(nn.Module):
    """
    Single camera 3D pose estimation network
    Takes 2D keypoints from single camera and lifts to 3D
    """
    
    def __init__(self, 
                 num_joints: int = 17,
                 hidden_dim: int = 1024,
                 num_blocks: int = 4,
                 dropout: float = 0.25,
                 use_camera_params: bool = True):
        super().__init__()
        
        self.num_joints = num_joints
        self.use_camera_params = use_camera_params
        
        # Input: 2D keypoints (x, y) + optional camera parameters
        input_dim = num_joints * 2
        if use_camera_params:
            input_dim += 9  # focal_x, focal_y, cx, cy, k1, k2, k3, p1, p2
        
        # Lifting network: 2D -> 3D
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        # Residual blocks
        for i in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))
        
        # Output layer: 3D coordinates (x, y, z) for each joint
        layers.append(nn.Linear(hidden_dim, num_joints * 3))
        
        self.lifting_net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, keypoints_2d: torch.Tensor, camera_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            keypoints_2d: (batch_size, num_joints, 2) - normalized 2D keypoints
            camera_params: (batch_size, 9) - camera intrinsics and distortion coefficients
        
        Returns:
            keypoints_3d: (batch_size, num_joints, 3) - 3D keypoints relative to root joint
        """
        batch_size = keypoints_2d.shape[0]
        
        # Flatten 2D keypoints
        kp_2d_flat = keypoints_2d.view(batch_size, -1)  # (B, num_joints * 2)
        
        # Concatenate camera parameters if available
        if self.use_camera_params and camera_params is not None:
            input_features = torch.cat([kp_2d_flat, camera_params], dim=1)
        else:
            input_features = kp_2d_flat
        
        # Lift to 3D
        keypoints_3d_flat = self.lifting_net(input_features)
        
        # Reshape to (batch_size, num_joints, 3)
        keypoints_3d = keypoints_3d_flat.view(batch_size, self.num_joints, 3)
        
        # Make root-relative (subtract hip/root joint position)
        root_joint = keypoints_3d[:, 0:1, :]  # Hip joint (index 0)
        keypoints_3d = keypoints_3d - root_joint
        
        return keypoints_3d

class CameraAwarePoseNet(nn.Module):
    """
    Enhanced version that explicitly handles camera parameters for ov9281
    """
    
    def __init__(self, 
                 num_joints: int = 17,
                 hidden_dim: int = 1024,
                 num_blocks: int = 4,
                 dropout: float = 0.25):
        super().__init__()
        
        self.num_joints = num_joints
        
        # Camera parameter processing branch
        self.camera_encoder = nn.Sequential(
            nn.Linear(9, 128),  # Camera intrinsics + distortion
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 2D keypoint processing branch
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(num_joints * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Combined processing
        combined_dim = 256 + 512  # camera + keypoints
        
        # Lifting network
        layers = []
        layers.append(nn.Linear(combined_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        for i in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_joints * 3))
        
        self.lifting_net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, keypoints_2d: torch.Tensor, camera_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with explicit camera handling
        
        Args:
            keypoints_2d: (batch_size, num_joints, 2)
            camera_params: (batch_size, 9) - [fx, fy, cx, cy, k1, k2, k3, p1, p2]
        
        Returns:
            keypoints_3d: (batch_size, num_joints, 3)
        """
        batch_size = keypoints_2d.shape[0]
        
        # Process camera parameters
        camera_features = self.camera_encoder(camera_params)  # (B, 256)
        
        # Process 2D keypoints
        kp_2d_flat = keypoints_2d.view(batch_size, -1)
        keypoint_features = self.keypoint_encoder(kp_2d_flat)  # (B, 512)
        
        # Combine features
        combined_features = torch.cat([camera_features, keypoint_features], dim=1)
        
        # Lift to 3D
        keypoints_3d_flat = self.lifting_net(combined_features)
        keypoints_3d = keypoints_3d_flat.view(batch_size, self.num_joints, 3)
        
        # Make root-relative
        root_joint = keypoints_3d[:, 0:1, :]
        keypoints_3d = keypoints_3d - root_joint
        
        return keypoints_3d

class PoseLoss(nn.Module):
    """Custom loss function for 3D pose estimation"""
    
    def __init__(self, 
                 joint_weights: Optional[torch.Tensor] = None,
                 bone_loss_weight: float = 0.1,
                 velocity_loss_weight: float = 0.01):
        super().__init__()
        
        self.joint_weights = joint_weights
        self.bone_loss_weight = bone_loss_weight
        self.velocity_loss_weight = velocity_loss_weight
        
        # Human3.6M bone connections for bone length consistency
        self.bones = [
            (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # legs
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine to head
            (8, 11), (11, 12), (12, 13),  # left arm
            (8, 14), (14, 15), (15, 16)   # right arm
        ]
    
    def forward(self, 
                pred_3d: torch.Tensor, 
                target_3d: torch.Tensor,
                pred_3d_prev: Optional[torch.Tensor] = None,
                target_3d_prev: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute pose estimation loss
        
        Args:
            pred_3d: (B, J, 3) predicted 3D poses
            target_3d: (B, J, 3) ground truth 3D poses
            pred_3d_prev: Previous frame predictions for temporal consistency
            target_3d_prev: Previous frame ground truth
        """
        # Joint position loss (L2)
        joint_loss = F.mse_loss(pred_3d, target_3d, reduction='none')
        
        # Apply joint weights if provided
        if self.joint_weights is not None:
            joint_weights = self.joint_weights.to(pred_3d.device).view(1, -1, 1)
            joint_loss = joint_loss * joint_weights
        
        joint_loss = joint_loss.mean()
        
        # Bone length consistency loss
        bone_loss = self._compute_bone_loss(pred_3d, target_3d)
        
        # Temporal consistency loss (if previous frame available)
        velocity_loss = torch.tensor(0.0, device=pred_3d.device)
        if pred_3d_prev is not None and target_3d_prev is not None:
            velocity_loss = self._compute_velocity_loss(pred_3d, target_3d, pred_3d_prev, target_3d_prev)
        
        # Total loss
        total_loss = joint_loss + self.bone_loss_weight * bone_loss + self.velocity_loss_weight * velocity_loss
        
        return {
            'total_loss': total_loss,
            'joint_loss': joint_loss,
            'bone_loss': bone_loss,
            'velocity_loss': velocity_loss
        }
    
    def _compute_bone_loss(self, pred_3d: torch.Tensor, target_3d: torch.Tensor) -> torch.Tensor:
        """Compute bone length consistency loss"""
        pred_bones = []
        target_bones = []
        
        for start_idx, end_idx in self.bones:
            if start_idx < pred_3d.shape[1] and end_idx < pred_3d.shape[1]:
                pred_bone = torch.norm(pred_3d[:, end_idx] - pred_3d[:, start_idx], dim=1)
                target_bone = torch.norm(target_3d[:, end_idx] - target_3d[:, start_idx], dim=1)
                
                pred_bones.append(pred_bone)
                target_bones.append(target_bone)
        
        if pred_bones:
            pred_bones = torch.stack(pred_bones, dim=1)  # (B, num_bones)
            target_bones = torch.stack(target_bones, dim=1)
            return F.mse_loss(pred_bones, target_bones)
        else:
            return torch.tensor(0.0, device=pred_3d.device)
    
    def _compute_velocity_loss(self, pred_3d: torch.Tensor, target_3d: torch.Tensor,
                             pred_3d_prev: torch.Tensor, target_3d_prev: torch.Tensor) -> torch.Tensor:
        """Compute temporal velocity consistency loss"""
        pred_velocity = pred_3d - pred_3d_prev
        target_velocity = target_3d - target_3d_prev
        return F.mse_loss(pred_velocity, target_velocity)


def create_ov9281_camera_params(image_width: int = 1280, image_height: int = 720) -> torch.Tensor:
    """
    Create default camera parameters for ov9281 160-degree camera
    You should replace these with your actual calibrated parameters
    """
    # Default parameters (you should calibrate your actual camera)
    focal_length = min(image_width, image_height) * 0.5  # Rough estimate for 160-degree FOV
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Default distortion coefficients (you should calibrate these)
    k1, k2, k3 = 0.0, 0.0, 0.0  # Radial distortion
    p1, p2 = 0.0, 0.0  # Tangential distortion
    
    camera_params = torch.tensor([
        focal_length, focal_length,  # fx, fy
        cx, cy,                      # cx, cy
        k1, k2, k3, p1, p2          # distortion coefficients
    ], dtype=torch.float32)
    
    return camera_params


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    simple_model = Monocular3DPoseNet().to(device)
    camera_aware_model = CameraAwarePoseNet().to(device)
    
    # Test data
    batch_size = 4
    num_joints = 17
    
    keypoints_2d = torch.randn(batch_size, num_joints, 2).to(device)
    camera_params = torch.randn(batch_size, 9).to(device)
    
    # Test forward pass
    with torch.no_grad():
        pred_3d_simple = simple_model(keypoints_2d, camera_params)
        pred_3d_camera = camera_aware_model(keypoints_2d, camera_params)
    
    print(f"Simple model output shape: {pred_3d_simple.shape}")
    print(f"Camera-aware model output shape: {pred_3d_camera.shape}")
    print(f"Models created successfully on {device}")