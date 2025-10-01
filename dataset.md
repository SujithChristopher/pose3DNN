# 3D Pose Estimation Datasets

This document provides a comprehensive list of datasets available for 3D human pose estimation from monocular camera images, including download links and key specifications.

## Latest Datasets (2024-2025)

### 1. WorldPose Dataset (2024)
- **Description**: World Cup dataset for global 3D human pose estimation in the wild
- **Scale**: 88 sequences, 2.5M+ 3D poses, 120km+ total travel distance
- **Coverage**: 1.75+ acres capture areas
- **Features**: Multi-person global pose estimation, FIFA World Cup footage
- **Download**: https://eth-ait.github.io/WorldPoseDataset/ (requires data request form)
- **License**: Research use only, FIFA footage requires additional agreement
- **Paper**: ECCV 2024

### 2. FreeMan Dataset (2024)
- **Description**: First large-scale real-world multi-view dataset captured by synchronizing 8 smartphones
- **Scale**: 11M frames from 8000 sequences across diverse scenarios
- **Features**: 8 synchronized smartphone views, diverse real-world scenarios
- **Download**: Available on Hugging Face & OpenXLab
- **License**: CC-BY-NC-4.0 (research purpose only)
- **Paper**: CVPR 2024

## Established 3D Pose Datasets

### 3. Human3.6M
- **Description**: Largest motion capture dataset with accurate 3D poses
- **Scale**: 3.6M human poses and corresponding images
- **Features**: 4 high-resolution cameras at 50Hz, controlled indoor environment
- **Subjects**: 11 professional actors
- **Activities**: 15 everyday activities (walking, eating, sitting, etc.)
- **Download**: Official website (requires registration and license agreement)
- **Website**: http://vision.imar.ro/human3.6m/

### 4. MPI-INF-3DHP
- **Description**: 3D human pose dataset with both indoor and outdoor scenes
- **Scale**: 1.3M+ frames from 14 camera views
- **Features**: 8 actors performing 8 activities, indoor/outdoor scenes
- **Evaluation**: Test set includes challenging outdoor scenarios
- **Download**: http://gvv.mpi-inf.mpg.de/3dhp-dataset/
- **Scripts**: Use get_dataset.sh (training) and get_testset.sh (test)

### 5. 3DPW (3D Poses in the Wild)
- **Description**: First in-the-wild dataset with accurate 3D poses for evaluation
- **Scale**: 60 video sequences captured with moving phone camera
- **Features**: IMU-based motion capture, challenging outdoor scenarios
- **Annotations**: 2D pose annotations, 3D poses, SMPL parameters
- **Download**: https://virtualhumans.mpi-inf.mpg.de/3DPW/
- **License**: Research use only

### 6. H3WB (Human3.6M 3D WholeBody)
- **Description**: Extension of Human3.6M with whole-body pose annotations
- **Scale**: 100K images with comprehensive annotations
- **Features**: 133 keypoints (17 body + 6 feet + 68 face + 42 hands)
- **Download**: https://github.com/wholebody3d/wholebody3d
- **Base**: Built on Human3.6M dataset

## Synthetic Datasets

### 7. SURREAL
- **Description**: Large-scale synthetic human dataset for pose estimation
- **Scale**: 6M frames with perfect ground truth annotations
- **Features**: Depth, body parts, optical flow, 2D/3D pose, surface normals
- **Rendering**: Realistic human models with diverse clothing and backgrounds
- **Download**: https://www.di.ens.fr/willow/research/surreal/data/
- **GitHub**: https://github.com/gulvarol/surreal
- **Size**: 86GB (excluding optical flow data)
- **License**: Requires acceptance of license terms

### 8. AGORA
- **Description**: Synthetic dataset with high realism and accurate ground truth
- **Features**: Registered 3D reference training data, rendered images
- **Evaluation**: Web-based evaluation platform
- **Download**: https://agora.is.tue.mpg.de/ (requires registration)
- **License**: Research use only

## Multi-View Datasets

### 9. CMU Panoptic
- **Description**: Multi-view dome capture system for social interactions
- **Scale**: 65 sequences (5.5 hours), 1.5M 3D skeletons
- **Setup**: 480 VGA cameras, 31 HD cameras in geodesic dome
- **Features**: Multi-person interactions, hand gestures, facial expressions
- **Download**: http://domedb.perception.cs.cmu.edu/
- **Toolbox**: https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
- **Script**: Use getData.sh for downloading
- **License**: Research use only, non-commercial

## 2D Pose Datasets (for 3D Lifting)

### 10. MPII Human Pose
- **Description**: State-of-the-art benchmark for 2D human pose estimation
- **Scale**: 25K images containing 40K+ people with annotated joints
- **Features**: 14 joint locations, body part occlusions, 3D torso orientations
- **Download Links**:
  - Images (12.9 GB): https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip
  - Annotations (12.5 MB): http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12.tar.gz
- **Website**: http://human-pose.mpi-inf.mpg.de/

### 11. LSP (Leeds Sports Pose)
- **Description**: Sports-focused pose dataset from Flickr
- **Scale**: 2000 annotated images with 14 joint locations
- **Features**: Sports activities, ~150 pixel person scale
- **Download Links**:
  - LSP Dataset: http://sam.johnson.io/research/lsp_dataset.zip
  - LSP Extended: http://sam.johnson.io/research/lspet_dataset.zip
- **Website**: http://sam.johnson.io/research/lsp.html

### 12. FLIC (Frames Labeled In Cinema)
- **Description**: Upper body pose dataset from Hollywood movies
- **Scale**: 5003 images from 30 popular movies
- **Features**: 10 upper body joints, movie frame extraction
- **Download**: https://bensapp.github.io/flic-dataset.html
- **TensorFlow**: Available through TF Datasets

### 13. COCO (Common Objects in Context)
- **Description**: Large-scale object detection dataset with pose annotations
- **Features**: 17 keypoints, diverse scenes and activities
- **Scale**: Hundreds of thousands of annotated images
- **Download**: Official COCO dataset website
- **Usage**: Commonly used for 2D-to-3D pose lifting

## Dataset Comparison

| Dataset | Type | Scale | Environment | Depth Info | License |
|---------|------|-------|-------------|------------|---------|
| WorldPose | Real | 2.5M poses | Outdoor | No | Research + FIFA |
| FreeMan | Real | 11M frames | Mixed | Multi-view | CC-BY-NC-4.0 |
| Human3.6M | Real | 3.6M poses | Indoor | MoCap | Research |
| MPI-INF-3DHP | Real | 1.3M frames | Mixed | MoCap | Research |
| 3DPW | Real | 60 sequences | Outdoor | IMU-based | Research |
| SURREAL | Synthetic | 6M frames | Synthetic | Perfect GT | License required |
| AGORA | Synthetic | Large | Synthetic | Perfect GT | Research |
| CMU Panoptic | Real | 1.5M skeletons | Indoor | Multi-view | Research |

## Usage Guidelines

### For Monocular 3D Pose Estimation:
1. **Training**: Use Human3.6M, MPI-INF-3DHP, SURREAL for large-scale training
2. **Evaluation**: Test on 3DPW for in-the-wild performance
3. **Pretraining**: Use COCO, MPII for 2D pose pretraining
4. **Synthetic Augmentation**: Use SURREAL, AGORA for data augmentation

### For Multi-Person Scenarios:
- WorldPose (latest, sports scenarios)
- MPI-INF-3DHP (outdoor scenes)
- CMU Panoptic (social interactions)

### For Whole-Body Pose:
- H3WB (face, hands, body, feet)
- CMU Panoptic (social interactions with gestures)

## Important Notes

### Licensing:
- Most datasets require registration and license acceptance
- Commercial use typically prohibited
- Proper citation required in research papers

### Data Access:
- Some datasets require institutional email for access
- Processing scripts often provided by dataset authors
- Large download sizes (several GB to TB)

### Preprocessing:
- Many datasets provide preprocessing scripts
- MMHuman3D framework supports multiple datasets
- Consider using established preprocessing pipelines

### Evaluation Metrics:
- MPJPE (Mean Per Joint Position Error)
- PA-MPJPE (Procrustes Aligned MPJPE)
- PCK (Percentage of Correct Keypoints)
- AUC (Area Under Curve)

## Citation Requirements

When using these datasets, ensure proper citation of:
1. Original dataset papers
2. Evaluation methodologies
3. Preprocessing frameworks (if used)
4. Baseline comparisons

## Additional Resources

- **MMHuman3D**: https://mmhuman3dyl-1993.readthedocs.io/en/latest/preprocess_dataset.html
- **Papers with Code**: https://paperswithcode.com/task/3d-pose-estimation
- **OpenPose**: https://github.com/CMU-Perceptual-Computing-Lab/openpose

---

**Last Updated**: September 2024
**Maintained by**: pose3DNN Project Team