"""
Pose to Stanford Indoor Map Alignment Module.
Aligns predicted PoseNet 3D camera position and orientation into the Stanford scene spatial coordinate frame.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mapping.coordinate_system import quaternion_to_rotation_matrix, get_camera_heading


class StanfordMapAligner:

    def __init__(self, scene_bounds=None):
        """
        Args:
            scene_bounds (dict, optional): Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.
        """
        self.bounds = scene_bounds

    def align_pose(self, predicted_position, predicted_quaternion):
        """
        Transform/align predicted camera pose into scene map coordinates.

        Returns:
            dict containing aligned 3D position, heading vector, and spatial validity check.
        """
        pos = np.array(predicted_position, dtype=np.float64)
        q = np.array(predicted_quaternion, dtype=np.float64)

        # Ensure quaternion normalization
        q = q / (np.linalg.norm(q) + 1e-8)

        # 3D Position
        x, y, z = pos

        # Heading vector in horizontal ground plane
        dx, dy = get_camera_heading(q)

        # Spatial bound check
        is_inside_bounds = True
        if self.bounds:
            if not (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                    self.bounds['y_min'] <= y <= self.bounds['y_max'] and
                    self.bounds['z_min'] <= z <= self.bounds['z_max']):
                is_inside_bounds = False

        aligned_pose = {
            "map_position": [float(x), float(y), float(z)],
            "heading_vector_2d": [float(dx), float(dy)],
            "quaternion": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            "is_valid_map_coordinate": is_inside_bounds
        }

        return aligned_pose


if __name__ == "__main__":
    aligner = StanfordMapAligner(scene_bounds={"x_min": -10.0, "x_max": 10.0, "y_min": -10.0, "y_max": 10.0, "z_min": 0.0, "z_max": 3.0})
    sample_pos = [1.25, 2.34, 1.50]
    sample_q = [0.99, 0.01, 0.02, 0.03]

    res = aligner.align_pose(sample_pos, sample_q)
    print("=" * 60)
    print("Stanford Map Pose Alignment Test:")
    print(f"Map Position: {res['map_position']}")
    print(f"2D Heading:   {res['heading_vector_2d']}")
    print(f"Inside Bounds: {res['is_valid_map_coordinate']}")
    print("=" * 60)
