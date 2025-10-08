"""
Interactive Marker Calibration Tool

This tool allows you to:
1. Click on markers in the camera image
2. Select corresponding mocap markers
3. Automatically calculate optimal rotation and translation parameters

Usage:
    python interactive_marker_calibration.py
"""

import numpy as np
import cv2
import msgpack as mp
import msgpack_numpy as mpn
import os
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import minimize
import pandas as pd


class InteractiveMarkerCalibration:
    def __init__(self, video_path, timestamp_path, mocap_data, marker_sets,
                 rvec, tvec, camera_matrix, dist_coeffs, R_m2w, frame_idx):
        """
        Interactive tool for calibrating marker transformations

        Parameters:
        -----------
        video_path : str
            Path to video file (msgpack)
        timestamp_path : str
            Path to timestamp file (msgpack)
        mocap_data : DataFrame
            Motion capture data
        marker_sets : dict
            Dictionary of marker point arrays
        rvec, tvec : ndarray
            Camera extrinsics
        camera_matrix, dist_coeffs : ndarray
            Camera intrinsics
        R_m2w : ndarray
            Rotation matrix from marker to world coordinates
        frame_idx : int
            Frame to use for calibration
        """
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.mocap_data = mocap_data
        self.marker_sets = marker_sets
        self.rvec = rvec
        self.tvec = tvec
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.R_m2w = R_m2w
        self.frame_idx = frame_idx

        # Current transformation parameters
        self.offset_x = 0
        self.offset_y = 0
        self.offset_3d_x = 0.0
        self.offset_3d_y = 0.0
        self.offset_3d_z = 0.0
        self.rotate_x = 0
        self.rotate_y = 0
        self.rotate_z = 0
        self.flip_x = True
        self.flip_y = False
        self.flip_z = True

        # Clicked points storage
        self.clicked_points_2d = []  # Points clicked in image
        self.selected_markers_3d = []  # Corresponding 3D marker positions
        self.marker_labels = []  # Labels for selected markers

        # Load frame
        self.frame = self._load_frame()
        self.display_frame = self.frame.copy()

        # Color scheme for markers
        self.colors = {
            'shoulder_left': (255, 0, 0),      # Red
            'shoulder_right': (0, 255, 0),     # Green
            'biceps_left': (0, 0, 255),        # Blue
            'biceps_right': (255, 255, 0),     # Cyan
            'wrist_left': (255, 0, 255),       # Magenta
            'wrist_right': (0, 255, 255),      # Yellow
            'trunk': (255, 255, 255)           # White
        }

        # Current mode
        self.mode = 'click'  # 'click' or 'adjust'
        self.selected_marker_set = None
        self.marker_list = self._build_marker_list()
        self.current_marker_idx = 0

    def _load_frame(self):
        """Load the calibration frame from video"""
        video_file = open(self.video_path, 'rb')
        video_data = mp.Unpacker(video_file, object_hook=mpn.decode)

        for i, _frame in enumerate(video_data):
            if i == self.frame_idx:
                frame = cv2.flip(_frame, 1)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                video_file.close()
                return frame

        video_file.close()
        raise ValueError(f"Could not load frame {self.frame_idx}")

    def _build_marker_list(self):
        """Build a flat list of all markers with their 3D positions"""
        marker_list = []
        for name, markers_3d in self.marker_sets.items():
            if len(markers_3d.shape) == 3:
                # Multiple markers
                for i in range(markers_3d.shape[0]):
                    point = markers_3d[i, self.frame_idx, :]
                    if not np.isnan(point).any():
                        marker_list.append({
                            'name': f"{name}_{i+1}",
                            'set': name,
                            'position': point,
                            'color': self.colors[name]
                        })
            else:
                # Single marker
                point = markers_3d[self.frame_idx, :]
                if not np.isnan(point).any():
                    marker_list.append({
                        'name': name,
                        'set': name,
                        'position': point,
                        'color': self.colors[name]
                    })

        return marker_list

    def _project_markers(self):
        """Project all mocap markers to 2D with current transformation"""
        projected_markers = []

        for marker in self.marker_list:
            # Get 3D position
            point_3d = marker['position'].copy()

            # Transform from marker to world coordinates
            point_3d_world = (self.R_m2w @ point_3d.reshape(3, 1)).flatten()

            # Apply flips
            if self.flip_x:
                point_3d_world[0] = -point_3d_world[0]
            if self.flip_y:
                point_3d_world[1] = -point_3d_world[1]
            if self.flip_z:
                point_3d_world[2] = -point_3d_world[2]

            # Apply rotation
            if self.rotate_x != 0 or self.rotate_y != 0 or self.rotate_z != 0:
                rotation = R_scipy.from_euler('xyz',
                    [self.rotate_x, self.rotate_y, self.rotate_z], degrees=True)
                point_3d_world = rotation.apply(point_3d_world)

            # Apply 3D offset
            point_3d_world[0] += self.offset_3d_x
            point_3d_world[1] += self.offset_3d_y
            point_3d_world[2] += self.offset_3d_z

            # Project to 2D
            point_2d, _ = cv2.fisheye.projectPoints(
                point_3d_world.reshape(1, 1, 3),
                self.rvec,
                self.tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            point_2d = point_2d.reshape(2)

            # Apply 2D offset
            point_2d[0] += self.offset_x
            point_2d[1] += self.offset_y

            projected_markers.append({
                'name': marker['name'],
                'position_2d': point_2d,
                'position_3d': point_3d_world,
                'color': marker['color']
            })

        return projected_markers

    def _draw_markers(self, frame, projected_markers, highlight_idx=None):
        """Draw projected markers on frame"""
        for i, marker in enumerate(projected_markers):
            x, y = int(marker['position_2d'][0]), int(marker['position_2d'][1])

            # Check if on screen
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                color = marker['color']

                # Highlight current marker
                if i == highlight_idx:
                    cv2.circle(frame, (x, y), 12, (0, 255, 255), 3)

                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)

                # Draw label
                cv2.putText(frame, marker['name'], (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_clicked_points(self, frame):
        """Draw user-clicked points"""
        for i, (pt, label) in enumerate(zip(self.clicked_points_2d, self.marker_labels)):
            cv2.circle(frame, tuple(pt), 8, (0, 255, 0), 2)
            cv2.putText(frame, f"{i+1}: {label}", (pt[0] + 15, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def _update_display(self):
        """Update the display frame"""
        self.display_frame = self.frame.copy()

        # Project and draw all markers
        projected_markers = self._project_markers()
        self._draw_markers(self.display_frame, projected_markers,
                          self.current_marker_idx if self.mode == 'click' else None)

        # Draw clicked points
        self._draw_clicked_points(self.display_frame)

        # Draw instructions
        instructions = []
        if self.mode == 'click':
            if len(self.clicked_points_2d) < len(self.marker_list):
                current_marker = self.marker_list[self.current_marker_idx]
                instructions.append(f"Mode: CLICK - Select marker: {current_marker['name']}")
                instructions.append("Click on the highlighted marker in the image")
                instructions.append("Press 'n' for next marker, 'p' for previous")
            else:
                instructions.append("All markers selected! Press 'c' to calculate optimal transform")
        else:
            instructions.append("Mode: ADJUST - Use keys to adjust transformation:")
            instructions.append("2D: Arrow keys (shift=fast) | 3D: W/S/A/D/Q/E")
            instructions.append("Rotate: R/T(X) F/G(Y) V/B(Z) | Flip: X/Y/Z keys")
            instructions.append("Press 'o' to optimize, 'r' to reset, 'c' to recalibrate")

        instructions.append(f"Correspondences: {len(self.clicked_points_2d)}")
        instructions.append("Press 's' to save, 'q' to quit")

        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(self.display_frame, text, (10, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show current parameters
        param_text = [
            f"2D: ({self.offset_x:.0f}, {self.offset_y:.0f}) px",
            f"3D: ({self.offset_3d_x:.3f}, {self.offset_3d_y:.3f}, {self.offset_3d_z:.3f}) m",
            f"Rot: ({self.rotate_x:.1f}, {self.rotate_y:.1f}, {self.rotate_z:.1f}) deg",
            f"Flip: X={self.flip_x} Y={self.flip_y} Z={self.flip_z}"
        ]

        y_offset = self.display_frame.shape[0] - 100
        for i, text in enumerate(param_text):
            cv2.putText(self.display_frame, text, (10, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN and self.mode == 'click':
            if self.current_marker_idx < len(self.marker_list):
                # Add clicked point
                self.clicked_points_2d.append([x, y])

                # Get corresponding 3D marker
                marker = self.marker_list[self.current_marker_idx]
                self.selected_markers_3d.append(marker['position'])
                self.marker_labels.append(marker['name'])

                print(f"Selected marker {marker['name']} at pixel ({x}, {y})")

                # Move to next marker
                self.current_marker_idx = min(self.current_marker_idx + 1,
                                             len(self.marker_list) - 1)

                self._update_display()

    def _optimize_transform(self):
        """Optimize transformation to minimize reprojection error"""
        if len(self.clicked_points_2d) < 3:
            print("Need at least 3 point correspondences for optimization")
            return

        print("\nOptimizing transformation parameters...")

        # Initial parameters: [offset_x, offset_y, offset_3d_x, offset_3d_y, offset_3d_z,
        #                      rotate_x, rotate_y, rotate_z]
        x0 = np.array([
            self.offset_x, self.offset_y,
            self.offset_3d_x, self.offset_3d_y, self.offset_3d_z,
            self.rotate_x, self.rotate_y, self.rotate_z
        ])

        def objective(params):
            """Compute reprojection error"""
            off_x, off_y, off_3dx, off_3dy, off_3dz, rot_x, rot_y, rot_z = params

            error = 0
            for i in range(len(self.clicked_points_2d)):
                # Transform 3D point
                point_3d = self.selected_markers_3d[i].copy()
                point_3d_world = (self.R_m2w @ point_3d.reshape(3, 1)).flatten()

                # Apply flips
                if self.flip_x:
                    point_3d_world[0] = -point_3d_world[0]
                if self.flip_y:
                    point_3d_world[1] = -point_3d_world[1]
                if self.flip_z:
                    point_3d_world[2] = -point_3d_world[2]

                # Apply rotation
                rotation = R_scipy.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
                point_3d_world = rotation.apply(point_3d_world)

                # Apply 3D offset
                point_3d_world[0] += off_3dx
                point_3d_world[1] += off_3dy
                point_3d_world[2] += off_3dz

                # Project to 2D
                point_2d, _ = cv2.fisheye.projectPoints(
                    point_3d_world.reshape(1, 1, 3),
                    self.rvec, self.tvec,
                    self.camera_matrix, self.dist_coeffs
                )
                point_2d = point_2d.reshape(2)

                # Apply 2D offset
                point_2d[0] += off_x
                point_2d[1] += off_y

                # Compute error
                clicked_pt = np.array(self.clicked_points_2d[i])
                error += np.sum((point_2d - clicked_pt) ** 2)

            return error

        # Optimize
        result = minimize(objective, x0, method='Powell')

        if result.success:
            # Update parameters
            self.offset_x, self.offset_y = result.x[0], result.x[1]
            self.offset_3d_x, self.offset_3d_y, self.offset_3d_z = result.x[2], result.x[3], result.x[4]
            self.rotate_x, self.rotate_y, self.rotate_z = result.x[5], result.x[6], result.x[7]

            print(f"Optimization successful! Final error: {result.fun:.2f}")
            print(f"2D offset: ({self.offset_x:.1f}, {self.offset_y:.1f}) px")
            print(f"3D offset: ({self.offset_3d_x:.3f}, {self.offset_3d_y:.3f}, {self.offset_3d_z:.3f}) m")
            print(f"Rotation: ({self.rotate_x:.2f}, {self.rotate_y:.2f}, {self.rotate_z:.2f}) deg")
        else:
            print("Optimization failed")

        self._update_display()

    def _save_parameters(self):
        """Save calibration parameters to file"""
        params = {
            'offset_x': float(self.offset_x),
            'offset_y': float(self.offset_y),
            'offset_3d_x': float(self.offset_3d_x),
            'offset_3d_y': float(self.offset_3d_y),
            'offset_3d_z': float(self.offset_3d_z),
            'rotate_x': float(self.rotate_x),
            'rotate_y': float(self.rotate_y),
            'rotate_z': float(self.rotate_z),
            'flip_x': bool(self.flip_x),
            'flip_y': bool(self.flip_y),
            'flip_z': bool(self.flip_z)
        }

        import json
        output_path = 'marker_calibration_params.json'
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"\nParameters saved to {output_path}")
        print("Copy these values to your notebook:")
        print(f"  offset_x = {self.offset_x:.1f}")
        print(f"  offset_y = {self.offset_y:.1f}")
        print(f"  offset_3d_x = {self.offset_3d_x:.3f}")
        print(f"  offset_3d_y = {self.offset_3d_y:.3f}")
        print(f"  offset_3d_z = {self.offset_3d_z:.3f}")
        print(f"  rotate_x = {self.rotate_x:.2f}")
        print(f"  rotate_y = {self.rotate_y:.2f}")
        print(f"  rotate_z = {self.rotate_z:.2f}")
        print(f"  flip_mocap_x = {self.flip_x}")
        print(f"  flip_mocap_y = {self.flip_y}")
        print(f"  flip_mocap_z = {self.flip_z}")

    def run(self):
        """Run the interactive calibration tool"""
        window_name = 'Marker Calibration Tool'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("Interactive Marker Calibration Tool")
        print("===================================")
        print("Instructions:")
        print("  1. Click on each highlighted marker in the image")
        print("  2. Press 'c' to calculate optimal transformation")
        print("  3. Press 's' to save parameters")
        print("  4. Press 'q' to quit")
        print("\nKeyboard shortcuts in ADJUST mode:")
        print("  Arrow keys: Adjust 2D offset")
        print("  W/S/A/D/Q/E: Adjust 3D offset")
        print("  R/T: Rotate X, F/G: Rotate Y, V/B: Rotate Z")
        print("  X/Y/Z: Toggle flip on respective axis")
        print("  O: Run optimization")

        self._update_display()

        while True:
            cv2.imshow(window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_parameters()
            elif key == ord('c'):
                if self.mode == 'click':
                    self._optimize_transform()
                    self.mode = 'adjust'
                else:
                    # Reset to click mode
                    self.clicked_points_2d = []
                    self.selected_markers_3d = []
                    self.marker_labels = []
                    self.current_marker_idx = 0
                    self.mode = 'click'
            elif key == ord('o'):
                self._optimize_transform()
            elif key == ord('r'):
                # Reset parameters
                self.offset_x = 0
                self.offset_y = 0
                self.offset_3d_x = 0.0
                self.offset_3d_y = 0.0
                self.offset_3d_z = 0.0
                self.rotate_x = 0
                self.rotate_y = 0
                self.rotate_z = 0
                self._update_display()
            elif key == ord('n') and self.mode == 'click':
                # Next marker
                self.current_marker_idx = min(self.current_marker_idx + 1,
                                             len(self.marker_list) - 1)
                self._update_display()
            elif key == ord('p') and self.mode == 'click':
                # Previous marker
                self.current_marker_idx = max(self.current_marker_idx - 1, 0)
                self._update_display()

            # Handle adjustment keys
            if self.mode == 'adjust':
                step = 10 if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) else 1

                # 2D offsets
                if key == 82:  # Up arrow
                    self.offset_y -= step
                    self._update_display()
                elif key == 84:  # Down arrow
                    self.offset_y += step
                    self._update_display()
                elif key == 81:  # Left arrow
                    self.offset_x -= step
                    self._update_display()
                elif key == 83:  # Right arrow
                    self.offset_x += step
                    self._update_display()

                # 3D offsets
                elif key == ord('w'):
                    self.offset_3d_y += 0.01
                    self._update_display()
                elif key == ord('s'):
                    self.offset_3d_y -= 0.01
                    self._update_display()
                elif key == ord('a'):
                    self.offset_3d_x -= 0.01
                    self._update_display()
                elif key == ord('d'):
                    self.offset_3d_x += 0.01
                    self._update_display()
                elif key == ord('q'):
                    self.offset_3d_z += 0.01
                    self._update_display()
                elif key == ord('e'):
                    self.offset_3d_z -= 0.01
                    self._update_display()

                # Rotations
                elif key == ord('r'):
                    self.rotate_x += 5
                    self._update_display()
                elif key == ord('t'):
                    self.rotate_x -= 5
                    self._update_display()
                elif key == ord('f'):
                    self.rotate_y += 5
                    self._update_display()
                elif key == ord('g'):
                    self.rotate_y -= 5
                    self._update_display()
                elif key == ord('v'):
                    self.rotate_z += 5
                    self._update_display()
                elif key == ord('b'):
                    self.rotate_z -= 5
                    self._update_display()

                # Flips
                elif key == ord('x'):
                    self.flip_x = not self.flip_x
                    self._update_display()
                elif key == ord('y'):
                    self.flip_y = not self.flip_y
                    self._update_display()
                elif key == ord('z'):
                    self.flip_z = not self.flip_z
                    self._update_display()

        cv2.destroyAllWindows()
        print("\nCalibration tool closed")


# Example usage function
def run_calibration(data_dir, recording_folder, calib_recording_folder, frame_idx=250):
    """
    Run the interactive calibration tool

    Parameters:
    -----------
    data_dir : str
        Parent directory containing MocapData
    recording_folder : str
        Recording folder name (e.g., 'yuv_t3')
    calib_recording_folder : str
        Calibration folder name (e.g., 'calib_yuvt3')
    frame_idx : int
        Frame to use for calibration (default: sync start frame)
    """
    import sys
    sys.path.append(os.path.dirname(__file__))
    from pd_support import read_df_csv, get_marker_name

    # Load data (same as in notebook)
    # data_dir should be the pose3DNN directory
    # MocapData is inside the pose3DNN directory
    data_path = os.path.join(data_dir, 'MocapData', recording_folder)
    calib_path = os.path.join(data_dir, 'MocapData', calib_recording_folder)

    print(f"Base directory: {data_dir}")
    print(f"Data path: {data_path}")
    print(f"Calib path: {calib_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Recording folder not found: {data_path}")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration folder not found: {calib_path}")

    # Load calibration data for R_m2w
    calib_mocap_file = os.path.join(calib_path, calib_recording_folder + '.csv')
    calib_mocap_data, _ = read_df_csv(calib_mocap_file)

    tr_m = get_marker_name('tr')
    tl_m = get_marker_name('tl')
    bl_m = get_marker_name('bl')
    br_m = get_marker_name('br')

    yvec = calib_mocap_data[[tr_m['x'], tr_m['y'], tr_m['z']]].values[0] - \
           calib_mocap_data[[br_m['x'], br_m['y'], br_m['z']]].values[0]
    xvec = calib_mocap_data[[br_m['x'], br_m['y'], br_m['z']]].values[0] - \
           calib_mocap_data[[bl_m['x'], bl_m['y'], bl_m['z']]].values[0]

    xq = xvec / np.linalg.norm(xvec)
    yvec = yvec - np.dot(yvec, xq) * xq
    yq = yvec / np.linalg.norm(yvec)
    zvec = np.cross(xq, yq)
    zq = zvec / np.linalg.norm(zvec)

    R_m2w = np.vstack([xq, yq, zq]).T

    # Load mocap data
    mocap_file = os.path.join(data_path, recording_folder + '.csv')
    mocap_data, start_time = read_df_csv(mocap_file)

    # Load camera calibration
    import toml
    calib_file = os.path.join(data_dir, 'notebooks', 'optimized_fisheye_calibration.toml')
    print(f"Camera calibration file: {calib_file}")
    calib_config = toml.load(calib_file)
    camera_matrix = np.array(calib_config['calibration']['camera_matrix'])
    dist_coeffs = np.array(calib_config['calibration']['dist_coeffs']).flatten()

    # Load rvec, tvec (you need to provide these from ArUco calibration)
    # For now, using placeholder - replace with actual values
    rvec = np.array([[-3.04800145], [0.18516046], [-0.61611019]])
    tvec = np.array([[0.03977474], [0.20073929], [0.84955837]])

    # Build marker sets
    def return_marker_points(mocap_data, markers):
        points = []
        for _m in markers:
            _marker_name = get_marker_name(_m)
            points.append(mocap_data[[_marker_name['x'], _marker_name['y'], _marker_name['z']]].values)
        return np.array(points).squeeze()

    shoulder_left = ['sr']
    wrist_left = ['flv1', 'flv2', 'flv3']
    biceps_left = ['blv1', 'blv2', 'blv3']
    shoulder_right = ['sl']
    wrist_right = ['frv1', 'frv2', 'frv3']
    biceps_right = ['brv1', 'brv2', 'brv3']
    trunk = ['tv1', 'tv2', 'tv3']

    marker_sets = {
        'shoulder_left': return_marker_points(mocap_data, shoulder_left),
        'shoulder_right': return_marker_points(mocap_data, shoulder_right),
        'biceps_left': return_marker_points(mocap_data, biceps_left),
        'biceps_right': return_marker_points(mocap_data, biceps_right),
        'wrist_left': return_marker_points(mocap_data, wrist_left),
        'wrist_right': return_marker_points(mocap_data, wrist_right),
        'trunk': return_marker_points(mocap_data, trunk)
    }

    video_path = os.path.join(data_path, 'webcam_color.msgpack')
    timestamp_path = os.path.join(data_path, 'webcam_timestamp.msgpack')

    # Create and run calibration tool
    tool = InteractiveMarkerCalibration(
        video_path, timestamp_path, mocap_data, marker_sets,
        rvec, tvec, camera_matrix, dist_coeffs, R_m2w, frame_idx
    )

    tool.run()


if __name__ == '__main__':
    # Example: Modify these paths to match your setup
    # Get the parent directory of the notebooks folder
    current_dir = os.path.dirname(os.path.abspath(__file__))  # notebooks directory
    parent_dir = os.path.dirname(current_dir)  # pose3DNN directory

    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")

    run_calibration(
        data_dir=parent_dir,
        recording_folder='yuv_t3',
        calib_recording_folder='calib_yuvt3',
        frame_idx=250
    )
