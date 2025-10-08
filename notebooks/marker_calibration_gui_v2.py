"""
Interactive Motion Capture Marker Calibration Tool V2 - PySide6 GUI
Features:
- Rotation and translation relative to marker-defined coordinate frames
- Selectable coordinate frame basis (trunk, biceps_left, biceps_right, wrist_left, wrist_right)
- Color-coded legend for marker visualization
- Real-time visual feedback
"""

import sys
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QSplitter, QScrollArea, QComboBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor

import msgpack as mp
import msgpack_numpy as mpn
import pandas as pd
from scipy.spatial.transform import Rotation as R


class MarkerCalibrationGUIv2(QMainWindow):
    """Main GUI window for interactive marker calibration with coordinate frame support"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Capture Marker Calibration Tool V2 - Coordinate Frame Based")
        self.setGeometry(100, 100, 1700, 950)

        # Data variables
        self.current_frame = None
        self.frame_with_markers = None
        self.frame_idx = 0
        self.max_frame_idx = 0

        # MoCap data
        self.mocap_data = None
        self.marker_sets = None
        self.marker_sets_raw = None  # Store raw 3-marker sets
        self.R_m2w = None

        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None

        # Video data
        self.video_frames = []

        # Color scheme for marker sets
        self.colors = {
            'shoulder_left': (255, 0, 0),      # Red
            'shoulder_right': (0, 255, 0),     # Green
            'biceps_left': (0, 0, 255),        # Blue
            'biceps_right': (255, 255, 0),     # Cyan
            'wrist_left': (255, 0, 255),       # Magenta
            'wrist_right': (0, 255, 255),      # Yellow
            'trunk': (255, 255, 255)           # White
        }

        # Coordinate frame basis options (3-marker sets only)
        self.coordinate_frame_options = ['trunk', 'biceps_left', 'biceps_right',
                                         'wrist_left', 'wrist_right']

        # Calibration parameters
        self.params = {
            'coordinate_frame': 'trunk',  # Which marker set to use as coordinate frame
            'offset_x': 210,
            'offset_y': 0,
            'offset_3d_x': -0.41,
            'offset_3d_y': 0.02,
            'offset_3d_z': 0.1,
            'rotate_x': 10,
            'rotate_y': 215,
            'rotate_z': -10,
            'flip_x': False,
            'flip_y': False,
            'flip_z': False
        }

        # Current coordinate frame transformation matrix
        self.R_frame = None  # Rotation from selected marker frame to world

        # Initialize UI
        self.init_ui()

        # Load data automatically
        self.auto_load_data()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Camera view
        left_panel = self.create_camera_panel()
        splitter.addWidget(left_panel)

        # Right panel: Controls
        right_panel = self.create_control_panel()
        splitter.addWidget(right_panel)

        # Set initial splitter sizes (65% camera, 35% controls)
        splitter.setSizes([1100, 600])

        main_layout.addWidget(splitter)

    def create_camera_panel(self):
        """Create the camera view panel with legend"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: #2b2b2b;")

        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Legend
        legend_group = self.create_legend()
        layout.addWidget(legend_group)

        # Frame control
        frame_control = QHBoxLayout()

        self.prev_frame_btn = QPushButton("← Previous")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        frame_control.addWidget(self.prev_frame_btn)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        frame_control.addWidget(self.frame_slider)

        self.next_frame_btn = QPushButton("Next →")
        self.next_frame_btn.clicked.connect(self.next_frame)
        frame_control.addWidget(self.next_frame_btn)

        self.frame_label = QLabel("Frame: 0 / 0")
        frame_control.addWidget(self.frame_label)

        layout.addLayout(frame_control)

        return panel

    def create_legend(self):
        """Create color-coded legend for marker sets"""
        group = QGroupBox("Marker Legend")
        layout = QHBoxLayout()

        for name, color in self.colors.items():
            # Create colored box
            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            r, g, b = color
            color_label.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid black;")

            # Create text label
            text_label = QLabel(name.replace('_', ' ').title())

            # Add to layout
            layout.addWidget(color_label)
            layout.addWidget(text_label)
            layout.addSpacing(10)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_control_panel(self):
        """Create the control panel with sliders"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Coordinate frame selection
        coord_frame_group = self.create_coordinate_frame_selector()
        scroll_layout.addWidget(coord_frame_group)

        # 2D Offset controls
        offset_2d_group = self.create_2d_offset_controls()
        scroll_layout.addWidget(offset_2d_group)

        # 3D Offset controls (relative to selected coordinate frame)
        offset_3d_group = self.create_3d_offset_controls()
        scroll_layout.addWidget(offset_3d_group)

        # Rotation controls (relative to selected coordinate frame)
        rotation_group = self.create_rotation_controls()
        scroll_layout.addWidget(rotation_group)

        # Flip controls
        flip_group = self.create_flip_controls()
        scroll_layout.addWidget(flip_group)

        # Action buttons
        button_layout = self.create_action_buttons()
        scroll_layout.addLayout(button_layout)

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def create_coordinate_frame_selector(self):
        """Create coordinate frame basis selector"""
        group = QGroupBox("Reference Coordinate Frame")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Select which 3-marker set defines the coordinate frame for rotations/translations:")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info_label)

        # Dropdown selector
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Basis:"))

        self.coord_frame_combo = QComboBox()
        for option in self.coordinate_frame_options:
            self.coord_frame_combo.addItem(option.replace('_', ' ').title(), option)
        self.coord_frame_combo.setCurrentText(self.params['coordinate_frame'].replace('_', ' ').title())
        self.coord_frame_combo.currentIndexChanged.connect(self.on_coordinate_frame_changed)
        frame_layout.addWidget(self.coord_frame_combo)

        layout.addLayout(frame_layout)

        # Display current frame info
        self.frame_info_label = QLabel()
        self.frame_info_label.setWordWrap(True)
        self.frame_info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
        layout.addWidget(self.frame_info_label)

        group.setLayout(layout)
        return group

    def create_2d_offset_controls(self):
        """Create 2D offset parameter controls"""
        group = QGroupBox("2D Image Offset (pixels)")
        layout = QVBoxLayout()

        # Offset X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Offset:"))
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-1000, 1000)
        self.offset_x_spin.setValue(self.params['offset_x'])
        self.offset_x_spin.valueChanged.connect(lambda v: self.update_param('offset_x', v))
        x_layout.addWidget(self.offset_x_spin)
        layout.addLayout(x_layout)

        # Offset Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Offset:"))
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-1000, 1000)
        self.offset_y_spin.setValue(self.params['offset_y'])
        self.offset_y_spin.valueChanged.connect(lambda v: self.update_param('offset_y', v))
        y_layout.addWidget(self.offset_y_spin)
        layout.addLayout(y_layout)

        group.setLayout(layout)
        return group

    def create_3d_offset_controls(self):
        """Create 3D offset parameter controls (relative to selected frame)"""
        group = QGroupBox("3D Offset (meters, relative to selected frame)")
        layout = QVBoxLayout()

        # Offset 3D X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Offset:"))
        self.offset_3d_x_spin = QDoubleSpinBox()
        self.offset_3d_x_spin.setRange(-5.0, 5.0)
        self.offset_3d_x_spin.setSingleStep(0.01)
        self.offset_3d_x_spin.setDecimals(3)
        self.offset_3d_x_spin.setValue(self.params['offset_3d_x'])
        self.offset_3d_x_spin.valueChanged.connect(lambda v: self.update_param('offset_3d_x', v))
        x_layout.addWidget(self.offset_3d_x_spin)
        layout.addLayout(x_layout)

        # Offset 3D Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Offset:"))
        self.offset_3d_y_spin = QDoubleSpinBox()
        self.offset_3d_y_spin.setRange(-5.0, 5.0)
        self.offset_3d_y_spin.setSingleStep(0.01)
        self.offset_3d_y_spin.setDecimals(3)
        self.offset_3d_y_spin.setValue(self.params['offset_3d_y'])
        self.offset_3d_y_spin.valueChanged.connect(lambda v: self.update_param('offset_3d_y', v))
        y_layout.addWidget(self.offset_3d_y_spin)
        layout.addLayout(y_layout)

        # Offset 3D Z
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z Offset:"))
        self.offset_3d_z_spin = QDoubleSpinBox()
        self.offset_3d_z_spin.setRange(-5.0, 5.0)
        self.offset_3d_z_spin.setSingleStep(0.01)
        self.offset_3d_z_spin.setDecimals(3)
        self.offset_3d_z_spin.setValue(self.params['offset_3d_z'])
        self.offset_3d_z_spin.valueChanged.connect(lambda v: self.update_param('offset_3d_z', v))
        z_layout.addWidget(self.offset_3d_z_spin)
        layout.addLayout(z_layout)

        group.setLayout(layout)
        return group

    def create_rotation_controls(self):
        """Create rotation parameter controls (relative to selected frame)"""
        group = QGroupBox("Rotation (degrees, relative to selected frame)")
        layout = QVBoxLayout()

        # Rotate X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("Pitch (X):"))
        self.rotate_x_spin = QDoubleSpinBox()
        self.rotate_x_spin.setRange(-180, 180)
        self.rotate_x_spin.setSingleStep(1)
        self.rotate_x_spin.setDecimals(1)
        self.rotate_x_spin.setValue(self.params['rotate_x'])
        self.rotate_x_spin.valueChanged.connect(lambda v: self.update_param('rotate_x', v))
        x_layout.addWidget(self.rotate_x_spin)
        layout.addLayout(x_layout)

        # Rotate Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Yaw (Y):"))
        self.rotate_y_spin = QDoubleSpinBox()
        self.rotate_y_spin.setRange(-180, 180)
        self.rotate_y_spin.setSingleStep(1)
        self.rotate_y_spin.setDecimals(1)
        self.rotate_y_spin.setValue(self.params['rotate_y'])
        self.rotate_y_spin.valueChanged.connect(lambda v: self.update_param('rotate_y', v))
        y_layout.addWidget(self.rotate_y_spin)
        layout.addLayout(y_layout)

        # Rotate Z
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Roll (Z):"))
        self.rotate_z_spin = QDoubleSpinBox()
        self.rotate_z_spin.setRange(-180, 180)
        self.rotate_z_spin.setSingleStep(1)
        self.rotate_z_spin.setDecimals(1)
        self.rotate_z_spin.setValue(self.params['rotate_z'])
        self.rotate_z_spin.valueChanged.connect(lambda v: self.update_param('rotate_z', v))
        z_layout.addWidget(self.rotate_z_spin)
        layout.addLayout(z_layout)

        group.setLayout(layout)
        return group

    def create_flip_controls(self):
        """Create axis flip controls"""
        group = QGroupBox("Axis Flips")
        layout = QVBoxLayout()

        self.flip_x_check = QCheckBox("Flip X (Left/Right)")
        self.flip_x_check.setChecked(self.params['flip_x'])
        self.flip_x_check.toggled.connect(lambda v: self.update_param('flip_x', v))
        layout.addWidget(self.flip_x_check)

        self.flip_y_check = QCheckBox("Flip Y (Up/Down)")
        self.flip_y_check.setChecked(self.params['flip_y'])
        self.flip_y_check.toggled.connect(lambda v: self.update_param('flip_y', v))
        layout.addWidget(self.flip_y_check)

        self.flip_z_check = QCheckBox("Flip Z (Front/Back)")
        self.flip_z_check.setChecked(self.params['flip_z'])
        self.flip_z_check.toggled.connect(lambda v: self.update_param('flip_z', v))
        layout.addWidget(self.flip_z_check)

        group.setLayout(layout)
        return group

    def create_action_buttons(self):
        """Create action buttons"""
        layout = QVBoxLayout()

        # Save button
        save_btn = QPushButton("Save Calibration")
        save_btn.clicked.connect(self.save_calibration)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        layout.addWidget(save_btn)

        # Load button
        load_btn = QPushButton("Load Calibration")
        load_btn.clicked.connect(self.load_calibration)
        load_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        layout.addWidget(load_btn)

        # Reset button
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_parameters)
        reset_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        layout.addWidget(reset_btn)

        return layout

    def compute_coordinate_frame(self, marker_set_name, frame_idx):
        """
        Compute coordinate frame from 3 markers using Gram-Schmidt orthogonalization

        Returns rotation matrix R_frame that transforms from marker-local to world coordinates
        """
        if marker_set_name not in self.marker_sets_raw:
            return np.eye(3)

        markers_3d = self.marker_sets_raw[marker_set_name]

        # Get 3 marker positions at current frame
        if len(markers_3d.shape) == 3 and markers_3d.shape[0] == 3:
            p1 = markers_3d[0, frame_idx, :]
            p2 = markers_3d[1, frame_idx, :]
            p3 = markers_3d[2, frame_idx, :]
        else:
            return np.eye(3)

        # Check for NaN values
        if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
            return np.eye(3)

        # Transform to world coordinates
        p1_world = (self.R_m2w @ p1.reshape(3, 1)).flatten()
        p2_world = (self.R_m2w @ p2.reshape(3, 1)).flatten()
        p3_world = (self.R_m2w @ p3.reshape(3, 1)).flatten()

        # Create coordinate frame using Gram-Schmidt
        # X-axis: from p1 to p2
        x_vec = p2_world - p1_world
        x_unit = x_vec / np.linalg.norm(x_vec)

        # Y-axis: orthogonalize (p3-p1) with respect to x_unit
        y_vec = p3_world - p1_world
        y_vec = y_vec - np.dot(y_vec, x_unit) * x_unit
        y_unit = y_vec / np.linalg.norm(y_vec)

        # Z-axis: cross product
        z_unit = np.cross(x_unit, y_unit)
        z_unit = z_unit / np.linalg.norm(z_unit)

        # Rotation matrix (columns are the frame axes)
        R_frame = np.vstack([x_unit, y_unit, z_unit]).T

        # Origin is at p1
        origin = p1_world

        return R_frame, origin

    def on_coordinate_frame_changed(self, index):
        """Handle coordinate frame selection change"""
        frame_name = self.coord_frame_combo.currentData()
        self.params['coordinate_frame'] = frame_name

        # Recompute coordinate frame
        if self.marker_sets_raw is not None and self.frame_idx < self.max_frame_idx:
            result = self.compute_coordinate_frame(frame_name, self.frame_idx)
            if isinstance(result, tuple):
                self.R_frame, self.frame_origin = result
            else:
                self.R_frame = result
                self.frame_origin = np.zeros(3)

            # Update info display
            self.update_frame_info()

        # Update display
        self.update_display()

    def update_frame_info(self):
        """Update coordinate frame info label"""
        frame_name = self.params['coordinate_frame']
        info_text = f"Using '{frame_name.replace('_', ' ').title()}' as coordinate frame basis.\n"
        info_text += f"Rotations and 3D translations are relative to this frame's local axes."
        self.frame_info_label.setText(info_text)

    def update_param(self, param_name, value):
        """Update a calibration parameter and refresh display"""
        self.params[param_name] = value
        self.update_display()

    def update_display(self):
        """Update the camera view with current parameters"""
        if self.current_frame is None or self.marker_sets is None:
            return

        # Draw markers on frame
        self.frame_with_markers = self.draw_markers_on_frame(
            self.current_frame,
            self.marker_sets,
            self.frame_idx,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs,
            self.R_m2w,
            R_frame=self.R_frame,
            frame_origin=self.frame_origin if hasattr(self, 'frame_origin') else np.zeros(3),
            offset_x=self.params['offset_x'],
            offset_y=self.params['offset_y'],
            offset_3d_x=self.params['offset_3d_x'],
            offset_3d_y=self.params['offset_3d_y'],
            offset_3d_z=self.params['offset_3d_z'],
            flip_mocap_x=self.params['flip_x'],
            flip_mocap_y=self.params['flip_y'],
            flip_mocap_z=self.params['flip_z'],
            rotate_x=self.params['rotate_x'],
            rotate_y=self.params['rotate_y'],
            rotate_z=self.params['rotate_z']
        )

        # Convert to QPixmap and display
        height, width, channel = self.frame_with_markers.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.frame_with_markers.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def on_frame_changed(self, value):
        """Handle frame slider change"""
        self.frame_idx = value
        self.frame_label.setText(f"Frame: {self.frame_idx} / {self.max_frame_idx}")

        if self.frame_idx < len(self.video_frames):
            self.current_frame = self.video_frames[self.frame_idx].copy()

            # Recompute coordinate frame for new frame
            frame_name = self.params['coordinate_frame']
            result = self.compute_coordinate_frame(frame_name, self.frame_idx)
            if isinstance(result, tuple):
                self.R_frame, self.frame_origin = result
            else:
                self.R_frame = result
                self.frame_origin = np.zeros(3)

            self.update_display()

    def prev_frame(self):
        """Go to previous frame"""
        if self.frame_idx > 0:
            self.frame_slider.setValue(self.frame_idx - 1)

    def next_frame(self):
        """Go to next frame"""
        if self.frame_idx < self.max_frame_idx:
            self.frame_slider.setValue(self.frame_idx + 1)

    def save_calibration(self):
        """Save calibration parameters to JSON file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            "marker_calibration_params.json",
            "JSON Files (*.json)"
        )

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.params, f, indent=4)

            print(f"Calibration saved to: {file_path}")
            print(f"  Coordinate Frame: {self.params['coordinate_frame']}")
            print(f"  2D Offset: ({self.params['offset_x']}, {self.params['offset_y']}) px")
            print(f"  3D Offset: ({self.params['offset_3d_x']:.3f}, {self.params['offset_3d_y']:.3f}, {self.params['offset_3d_z']:.3f}) m")
            print(f"  Rotation: ({self.params['rotate_x']:.1f}, {self.params['rotate_y']:.1f}, {self.params['rotate_z']:.1f}) deg")

    def load_calibration(self):
        """Load calibration parameters from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            "",
            "JSON Files (*.json)"
        )

        if file_path:
            with open(file_path, 'r') as f:
                loaded_params = json.load(f)

            # Update coordinate frame selector
            if 'coordinate_frame' in loaded_params:
                frame_name = loaded_params['coordinate_frame']
                index = self.coord_frame_combo.findData(frame_name)
                if index >= 0:
                    self.coord_frame_combo.setCurrentIndex(index)

            # Update UI controls
            self.offset_x_spin.setValue(loaded_params.get('offset_x', 0))
            self.offset_y_spin.setValue(loaded_params.get('offset_y', 0))
            self.offset_3d_x_spin.setValue(loaded_params.get('offset_3d_x', 0))
            self.offset_3d_y_spin.setValue(loaded_params.get('offset_3d_y', 0))
            self.offset_3d_z_spin.setValue(loaded_params.get('offset_3d_z', 0))
            self.rotate_x_spin.setValue(loaded_params.get('rotate_x', 0))
            self.rotate_y_spin.setValue(loaded_params.get('rotate_y', 0))
            self.rotate_z_spin.setValue(loaded_params.get('rotate_z', 0))
            self.flip_x_check.setChecked(loaded_params.get('flip_x', False))
            self.flip_y_check.setChecked(loaded_params.get('flip_y', False))
            self.flip_z_check.setChecked(loaded_params.get('flip_z', False))

            print(f"Calibration loaded from: {file_path}")

    def reset_parameters(self):
        """Reset all parameters to default values"""
        defaults = {
            'coordinate_frame': 'trunk',
            'offset_x': 210,
            'offset_y': 0,
            'offset_3d_x': -0.41,
            'offset_3d_y': 0.02,
            'offset_3d_z': 0.1,
            'rotate_x': 10,
            'rotate_y': 215,
            'rotate_z': -10,
            'flip_x': False,
            'flip_y': False,
            'flip_z': False
        }

        # Update coordinate frame
        index = self.coord_frame_combo.findData(defaults['coordinate_frame'])
        if index >= 0:
            self.coord_frame_combo.setCurrentIndex(index)

        self.offset_x_spin.setValue(defaults['offset_x'])
        self.offset_y_spin.setValue(defaults['offset_y'])
        self.offset_3d_x_spin.setValue(defaults['offset_3d_x'])
        self.offset_3d_y_spin.setValue(defaults['offset_3d_y'])
        self.offset_3d_z_spin.setValue(defaults['offset_3d_z'])
        self.rotate_x_spin.setValue(defaults['rotate_x'])
        self.rotate_y_spin.setValue(defaults['rotate_y'])
        self.rotate_z_spin.setValue(defaults['rotate_z'])
        self.flip_x_check.setChecked(defaults['flip_x'])
        self.flip_y_check.setChecked(defaults['flip_y'])
        self.flip_z_check.setChecked(defaults['flip_z'])

    def auto_load_data(self):
        """Automatically load data from default paths"""
        # Get paths
        notebook_dir = Path(__file__).parent
        parent_dir = notebook_dir.parent
        mocap_data_dir = parent_dir / 'MocapData'

        # Load from yuv_t3 recording
        recording_folder = 'yuv_t3'
        calib_recording_folder = 'calib_yuvt3'

        data_dir = mocap_data_dir / recording_folder
        calib_data_dir = mocap_data_dir / calib_recording_folder

        try:
            # Load calibration data (R_m2w matrix)
            self.R_m2w = self.load_calibration_transform(calib_data_dir, calib_recording_folder)

            # Load camera calibration
            self.load_camera_calibration(notebook_dir)

            # Load camera extrinsics
            self.load_camera_extrinsics(calib_data_dir, calib_recording_folder)

            # Load MoCap data
            self.load_mocap_data(data_dir, recording_folder)

            # Load video frames
            self.load_video_frames(data_dir)

            # Initialize coordinate frame
            result = self.compute_coordinate_frame(self.params['coordinate_frame'], self.frame_idx)
            if isinstance(result, tuple):
                self.R_frame, self.frame_origin = result
            else:
                self.R_frame = result
                self.frame_origin = np.zeros(3)

            self.update_frame_info()

            print("All data loaded successfully!")

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()

    def load_calibration_transform(self, calib_data_dir, calib_recording_folder):
        """Load the R_m2w transformation matrix from calibration data"""
        import sys
        sys.path.append(str(Path(__file__).parent))
        from pd_support import read_df_csv, get_marker_name

        mocap_file = calib_data_dir / f'{calib_recording_folder}.csv'
        mocap_data, _ = read_df_csv(str(mocap_file))

        tr_m = get_marker_name('tr')
        tl_m = get_marker_name('tl')
        bl_m = get_marker_name('bl')
        br_m = get_marker_name('br')

        yvec = mocap_data[[tr_m['x'], tr_m['y'], tr_m['z']]].values[0] - mocap_data[[br_m['x'], br_m['y'], br_m['z']]].values[0]
        xvec = mocap_data[[br_m['x'], br_m['y'], br_m['z']]].values[0] - mocap_data[[bl_m['x'], bl_m['y'], bl_m['z']]].values[0]

        # Gram-Schmidt process
        xq = xvec / np.linalg.norm(xvec)
        yvec = yvec - np.dot(yvec, xq) * xq
        yq = yvec / np.linalg.norm(yvec)
        zvec = np.cross(xq, yq)
        zq = zvec / np.linalg.norm(zvec)

        R_m2w = np.vstack([xq, yq, zq]).T

        return R_m2w

    def load_camera_calibration(self, notebook_dir):
        """Load camera intrinsics from TOML file"""
        import toml

        calib_file = notebook_dir / 'optimized_fisheye_calibration.toml'
        calib_config = toml.load(str(calib_file))

        self.camera_matrix = np.array(calib_config['calibration']['camera_matrix'])
        self.dist_coeffs = np.array(calib_config['calibration']['dist_coeffs']).flatten()

    def load_camera_extrinsics(self, calib_data_dir, calib_recording_folder):
        """Load camera extrinsics (rvec, tvec) using ArUco markers"""
        import sys
        sys.path.append(str(Path(__file__).parent))
        from cv2 import aruco
        from pd_support import read_df_csv, get_marker_name

        # Load calibration frame
        calib_video_file = open(calib_data_dir / 'webcam_color.msgpack', 'rb')
        calib_video_data = mp.Unpacker(calib_video_file, object_hook=mpn.decode)

        calib_frame = None
        for _frame in calib_video_data:
            calib_frame = cv2.flip(_frame, 1)
            break

        calib_video_file.close()

        # Detect ArUco markers
        dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dict_aruco, parameters)
        corners, ids, _ = detector.detectMarkers(calib_frame)

        # Get MoCap corner positions
        mocap_file = calib_data_dir / f'{calib_recording_folder}.csv'
        mocap_data, _ = read_df_csv(str(mocap_file))

        tr_m = get_marker_name('tr')
        tl_m = get_marker_name('tl')
        bl_m = get_marker_name('bl')
        br_m = get_marker_name('br')

        tr_pos = mocap_data[[tr_m['x'], tr_m['y'], tr_m['z']]].values[0]
        tl_pos = mocap_data[[tl_m['x'], tl_m['y'], tl_m['z']]].values[0]
        bl_pos = mocap_data[[bl_m['x'], bl_m['y'], bl_m['z']]].values[0]
        br_pos = mocap_data[[br_m['x'], br_m['y'], br_m['z']]].values[0]

        corners_marker = np.array([tr_pos, tl_pos, bl_pos, br_pos])
        corners_world = (self.R_m2w @ corners_marker.T).T

        # Solve PnP
        detected_corners_2d = corners[1][0]
        objPoints_mocap = np.array([
            corners_world[1],  # TL
            corners_world[0],  # TR
            corners_world[3],  # BR
            corners_world[2]   # BL
        ], dtype=np.float32)

        success, self.rvec, self.tvec = cv2.solvePnP(
            objPoints_mocap,
            detected_corners_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    def load_mocap_data(self, data_dir, recording_folder):
        """Load motion capture marker data"""
        import sys
        sys.path.append(str(Path(__file__).parent))
        from pd_support import read_df_csv, get_marker_name

        mocap_file = data_dir / f'{recording_folder}.csv'
        self.mocap_data, _ = read_df_csv(str(mocap_file))

        # Extract marker sets
        def return_marker_points(mocap_data, markers):
            points = []
            _marker_name = []
            for _m in markers:
                _marker_name.append(get_marker_name(_m))
            for m in _marker_name:
                points.append(mocap_data[[m['x'], m['y'], m['z']]].values)
            return np.array(points).squeeze()

        shoulder_left = ['sr']
        wrist_left = ['flv1', 'flv2', 'flv3']
        biceps_left = ['blv1', 'blv2', 'blv3']
        shoulder_right = ['sl']
        wrist_right = ['frv1', 'frv2', 'frv3']
        biceps_right = ['brv1', 'brv2', 'brv3']
        trunk = ['tv1', 'tv2', 'tv3']

        # Store raw 3-marker sets for coordinate frame computation
        self.marker_sets_raw = {
            'biceps_left': return_marker_points(self.mocap_data, biceps_left),
            'biceps_right': return_marker_points(self.mocap_data, biceps_right),
            'wrist_left': return_marker_points(self.mocap_data, wrist_left),
            'wrist_right': return_marker_points(self.mocap_data, wrist_right),
            'trunk': return_marker_points(self.mocap_data, trunk)
        }

        # Store all marker sets for visualization
        self.marker_sets = {
            'shoulder_left': return_marker_points(self.mocap_data, shoulder_left),
            'shoulder_right': return_marker_points(self.mocap_data, shoulder_right),
            'biceps_left': self.marker_sets_raw['biceps_left'],
            'biceps_right': self.marker_sets_raw['biceps_right'],
            'wrist_left': self.marker_sets_raw['wrist_left'],
            'wrist_right': self.marker_sets_raw['wrist_right'],
            'trunk': self.marker_sets_raw['trunk']
        }

    def load_video_frames(self, data_dir):
        """Load video frames into memory"""
        video_file_path = data_dir / 'webcam_color.msgpack'
        timestamp_file_path = data_dir / 'webcam_timestamp.msgpack'

        # Load timestamps to find sync start
        with open(timestamp_file_path, 'rb') as f:
            timestamp_data = mp.Unpacker(f, object_hook=mpn.decode)
            start_sync = None
            for i, ts in enumerate(timestamp_data):
                _sync = ts[0]
                if _sync == 1 and start_sync is None:
                    start_sync = i
                    break

        if start_sync is None:
            start_sync = 0

        # Load frames starting from sync point
        with open(video_file_path, 'rb') as f:
            video_data = mp.Unpacker(f, object_hook=mpn.decode)

            for i, _frame in enumerate(video_data):
                if i < start_sync:
                    continue

                frame = cv2.flip(_frame, 1)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                self.video_frames.append(frame)

                # Load only first 300 frames to save memory
                if len(self.video_frames) >= 300:
                    break

        self.max_frame_idx = len(self.video_frames) - 1
        self.frame_slider.setMaximum(self.max_frame_idx)
        self.frame_idx = 0

        if len(self.video_frames) > 0:
            self.current_frame = self.video_frames[0].copy()
            self.update_display()

    def project_mocap_to_image(self, mocap_points_3d, rvec, tvec, camera_matrix, dist_coeffs,
                               R_frame, frame_origin,
                               flip_horizontal=False, image_width=1200,
                               offset_x=0, offset_y=0,
                               offset_3d_x=0, offset_3d_y=0, offset_3d_z=0,
                               flip_mocap_x=False, flip_mocap_y=False, flip_mocap_z=False,
                               rotate_x=0, rotate_y=0, rotate_z=0):
        """Project 3D MoCap points to 2D image coordinates with frame-relative transforms"""
        if len(mocap_points_3d.shape) == 1:
            mocap_points_3d = mocap_points_3d.reshape(1, 3)

        valid_mask = ~np.isnan(mocap_points_3d).any(axis=1)

        if not valid_mask.any():
            return np.array([]).reshape(0, 2), valid_mask

        mocap_points_3d_valid = mocap_points_3d[valid_mask].copy()

        # Apply flips in world coordinates
        if flip_mocap_x:
            mocap_points_3d_valid[:, 0] = -mocap_points_3d_valid[:, 0]
        if flip_mocap_y:
            mocap_points_3d_valid[:, 1] = -mocap_points_3d_valid[:, 1]
        if flip_mocap_z:
            mocap_points_3d_valid[:, 2] = -mocap_points_3d_valid[:, 2]

        # Transform points to coordinate frame local space
        # Points are in world coords, convert to frame-local coords
        R_frame_inv = R_frame.T  # Inverse rotation
        points_local = np.zeros_like(mocap_points_3d_valid)
        for i in range(len(mocap_points_3d_valid)):
            # Translate to frame origin, then rotate to local coordinates
            point_translated = mocap_points_3d_valid[i] - frame_origin
            points_local[i] = R_frame_inv @ point_translated

        # Apply 3D offset in local frame coordinates
        points_local[:, 0] += offset_3d_x
        points_local[:, 1] += offset_3d_y
        points_local[:, 2] += offset_3d_z

        # Apply rotation in local frame coordinates
        if rotate_x != 0 or rotate_y != 0 or rotate_z != 0:
            rotation = R.from_euler('xyz', [rotate_x, rotate_y, rotate_z], degrees=True)
            rotation_matrix = rotation.as_matrix()
            points_local = (rotation_matrix @ points_local.T).T

        # Transform back to world coordinates
        points_world = np.zeros_like(points_local)
        for i in range(len(points_local)):
            points_world[i] = R_frame @ points_local[i] + frame_origin

        # Project points
        points_2d, _ = cv2.fisheye.projectPoints(
            points_world.reshape(-1, 1, 3),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        points_2d = points_2d.reshape(-1, 2)

        if flip_horizontal:
            points_2d[:, 0] = image_width - points_2d[:, 0]

        points_2d[:, 0] += offset_x
        points_2d[:, 1] += offset_y

        return points_2d, valid_mask

    def draw_markers_on_frame(self, frame, marker_sets, frame_idx, rvec, tvec,
                              camera_matrix, dist_coeffs, R_m2w, R_frame, frame_origin,
                              offset_x=0, offset_y=0,
                              offset_3d_x=0, offset_3d_y=0, offset_3d_z=0,
                              flip_mocap_x=False, flip_mocap_y=False, flip_mocap_z=False,
                              rotate_x=0, rotate_y=0, rotate_z=0):
        """Draw all MoCap markers on a video frame"""
        frame_draw = frame.copy()

        for name, markers_3d in marker_sets.items():
            if len(markers_3d.shape) == 3:
                points_3d_marker = markers_3d[:, frame_idx, :]
            else:
                points_3d_marker = markers_3d[frame_idx, :].reshape(1, 3)

            points_3d_world = (R_m2w @ points_3d_marker.T).T

            points_2d, valid_mask = self.project_mocap_to_image(
                points_3d_world, rvec, tvec, camera_matrix, dist_coeffs,
                R_frame, frame_origin,
                flip_horizontal=False, image_width=frame.shape[1],
                offset_x=offset_x, offset_y=offset_y,
                offset_3d_x=offset_3d_x, offset_3d_y=offset_3d_y, offset_3d_z=offset_3d_z,
                flip_mocap_x=flip_mocap_x, flip_mocap_y=flip_mocap_y, flip_mocap_z=flip_mocap_z,
                rotate_x=rotate_x, rotate_y=rotate_y, rotate_z=rotate_z
            )

            if len(points_2d) == 0:
                continue

            color = self.colors.get(name, (128, 128, 128))
            for pt in points_2d:
                if not np.isnan(pt).any():
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(frame_draw, (x, y), 5, color, -1)
                    cv2.circle(frame_draw, (x, y), 6, (0, 0, 0), 2)

            if len(points_2d) > 1:
                for i in range(len(points_2d) - 1):
                    if not np.isnan(points_2d[i]).any() and not np.isnan(points_2d[i+1]).any():
                        pt1 = tuple(points_2d[i].astype(int))
                        pt2 = tuple(points_2d[i+1].astype(int))
                        cv2.line(frame_draw, pt1, pt2, color, 2)

        return frame_draw


def main():
    app = QApplication(sys.argv)
    window = MarkerCalibrationGUIv2()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
