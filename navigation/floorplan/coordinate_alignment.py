"""
coordinate_alignment.py - Coordinate Frame Registration & Alignment

Provides transformation framework mapping between:
COLMAP / 7-Scenes 3D camera world coordinates -> Indoor floor-plan coordinates.

Supports rigid / similarity transformations:
P_floorplan = s * R * P_world + T
"""

import math
import numpy as np


class CoordinateAlignmentManager:
    def __init__(self, scale=1.0, translation=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0)):
        """
        Configurable transformation parameters.
        scale: float
        translation: (tx, ty, tz)
        rotation_deg: (rx, ry, rz) Euler angles in degrees
        """
        self.scale = float(scale)
        self.translation = np.array(translation, dtype=np.float64)
        self.rotation_deg = rotation_deg
        self.R = self._compute_rotation_matrix(rotation_deg)

    def _compute_rotation_matrix(self, degrees):
        rx, ry, rz = [math.radians(d) for d in degrees]

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def world_to_floorplan(self, pos_xyz):
        """
        Transforms a 3D position [x, y, z] from 7-Scenes world coordinates into floor-plan coordinates.
        """
        P = np.array(pos_xyz, dtype=np.float64)
        P_transformed = self.scale * (self.R @ P) + self.translation
        return P_transformed.tolist()

    def floorplan_to_world(self, pos_xyz):
        """
        Inverse transformation from floor-plan coordinates to 7-Scenes world coordinates.
        """
        P = np.array(pos_xyz, dtype=np.float64)
        P_untranslated = (P - self.translation) / self.scale
        P_world = self.R.T @ P_untranslated
        return P_world.tolist()


if __name__ == "__main__":
    aligner = CoordinateAlignmentManager(scale=1.0, translation=(0.5, -0.2, 0.0), rotation_deg=(0, 0, 15))
    sample_pt = [-0.12, -1.12, -0.98]
    transformed = aligner.world_to_floorplan(sample_pt)
    recovered = aligner.floorplan_to_world(transformed)

    print("==================================================")
    print("Coordinate Alignment Transformation Test")
    print("==================================================")
    print(f"Sample World Pt:       {sample_pt}")
    print(f"Floorplan Transformed: {[round(v, 4) for v in transformed]}")
    print(f"Recovered World Pt:    {[round(v, 4) for v in recovered]}")
    print("==================================================")
