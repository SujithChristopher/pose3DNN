# 3D Pose Estimation for OV9281 Camera

A PyTorch-based 3D human pose estimation system designed for monocular OV9281 160-degree cameras.

## Features

- **Single camera 3D pose estimation** - Lifts 2D keypoints to 3D poses
- **OV9281 camera support** - Optimized for 160-degree FOV monochrome cameras
- **Camera calibration utilities** - Handle distortion correction and parameter management
- **Synthetic dataset** - Quick start with Human3.6M-style synthetic data
- **Real-time inference** - Process video streams and live camera feeds
- **Two model architectures** - Simple lifting network and camera-aware model

## Quick Start

### 1. Setup Environment

```bash
# Activate your conda environment (r12 in this case)
conda activate r12

# Install required packages if needed
pip install torch torchvision opencv-python matplotlib tqdm scipy scikit-learn
```

### 2. Run Demo

```bash
# Run complete demo (dataset, model, training, inference)
python quick_start.py

# Or run specific components
python quick_start.py --demo dataset    # Test dataset creation
python quick_start.py --demo model      # Test model creation
python quick_start.py --demo train      # Quick training (5 epochs)
python quick_start.py --demo inference  # Test inference
```

### 3. Train Your Model

```bash
# Train simple model for 50 epochs
python train.py --model simple --epochs 50 --batch_size 8

# Train camera-aware model
python train.py --model camera_aware --epochs 50 --batch_size 8

# Resume training from checkpoint
python train.py --resume ./checkpoints/best_model.pth
```

### 4. Run Inference

```bash
# Process single image
python inference.py --model_path ./checkpoints/best_model.pth --input image.jpg

# Process video
python inference.py --model_path ./checkpoints/best_model.pth --input video.mp4 --output output.mp4

# Live camera inference
python inference.py --model_path ./checkpoints/best_model.pth --input camera --camera_id 0
```

## File Structure

```
poseNN/
├── dataset_loader.py      # Dataset creation and loading
├── model.py              # Neural network architectures
├── camera_utils.py       # OV9281 camera calibration utilities
├── train.py             # Training script
├── inference.py         # Real-time inference
├── quick_start.py       # Demo and testing script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Model Architectures

### 1. Simple Lifting Network (`Monocular3DPoseNet`)
- Takes 2D keypoints + camera parameters
- Residual blocks for feature extraction
- Direct lifting to 3D coordinates
- ~8.5M parameters

### 2. Camera-Aware Network (`CameraAwarePoseNet`) 
- Separate processing branches for camera and keypoints
- Explicit camera parameter encoding
- Better handling of camera-specific distortions
- ~8.5M parameters

## Camera Setup

### Setting Your Camera Parameters

1. **Use default parameters** (for quick testing):
```python
from camera_utils import create_sample_camera_params
camera_matrix, dist_coeffs = create_sample_camera_params()
```

2. **Calibrate your camera** (recommended):
```python
from camera_utils import calibrate_camera_from_chessboard
camera_matrix, dist_coeffs = calibrate_camera_from_chessboard(
    images_path="./calibration_images/",
    chessboard_size=(9, 6),
    square_size=25.0  # mm
)
```

3. **Set known parameters** (if you have them):
```python
from camera_utils import OV9281CameraCalibration
import numpy as np

camera_matrix = np.array([
    [focal_x, 0, cx],
    [0, focal_y, cy], 
    [0, 0, 1]
])
dist_coeffs = np.array([k1, k2, k3, p1, p2])

calibration = OV9281CameraCalibration(
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)
```

### OV9281 Typical Parameters
For 160-degree FOV at 1280x720:
- **Focal length**: ~180-250 pixels (very wide FOV)
- **Principal point**: (640, 360) - center of image
- **Distortion**: Significant barrel distortion expected

## Training Data

The current implementation uses synthetic Human3.6M-style data for quick prototyping. For production use:

1. **Integrate real 2D pose detection**:
   - MediaPipe Pose
   - OpenPose
   - AlphaPose
   - PoseNet

2. **Use real datasets**:
   - Human3.6M
   - MPII-3DHP
   - MPI-INF-3DHP
   - 3DPW

3. **Collect your own data**:
   - Multi-view camera setup for ground truth
   - Motion capture system
   - Depth cameras for validation

## Loss Functions

- **Joint position loss**: L2 distance between predicted and ground truth 3D joints
- **Bone length consistency**: Enforces anatomically correct bone lengths
- **Temporal smoothness**: Optional velocity consistency across frames

## Performance Metrics

- **MPJPE**: Mean Per Joint Position Error (mm)
- **PA-MPJPE**: Procrustes Aligned MPJPE (removes global rotation/translation)
- **PCK**: Percentage of Correct Keypoints
- **AUC**: Area Under Curve for PCK

## Limitations & Next Steps

### Current Limitations
1. **Synthetic data only** - needs real training data
2. **Dummy 2D pose detection** - integrate real 2D detector  
3. **Basic camera model** - could add more sophisticated camera handling
4. **No temporal modeling** - single frame processing only

### Recommended Improvements
1. **Add real 2D pose detector** (MediaPipe recommended)
2. **Collect multi-view training data** for your specific use case
3. **Add temporal smoothing** for video sequences
4. **Implement multi-person pose estimation**
5. **Add pose validation** and outlier detection
6. **Optimize for real-time performance** on edge devices

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (tested with current setup)
- **RAM**: 8GB+ for training
- **Storage**: 1GB+ for models and data
- **Camera**: OV9281 or similar monochrome camera

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ov9281_pose_estimation,
  title={3D Pose Estimation for OV9281 Camera},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/poseNN}
}
```