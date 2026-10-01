"""
Stanford 2D-3D-Semantics Coordinate System Specification & Transformations.

CONVENTIONS:
1. World Coordinate System: Right-handed 3D spatial coordinate frame.
   - X axis: Horizontal ground dimension (meters)
   - Y axis: Depth / horizontal ground dimension (meters)
   - Z axis: Elevation / height above floor level (meters)
2. Camera Pose Convention:
   - Position: [x, y, z] camera focal center in world coordinates (meters).
   - Quaternion: [qw, qx, qy, qz] representing camera rotation relative to world axes.
   - Quaternion is unit-length: ||q|| = 1.0.
"""

import numpy as np


def quaternion_to_rotation_matrix(q):
    """
    Convert a normalized quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.
    """
    q = np.array(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-8)
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,      2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,      1 - 2*qx**2 - 2*qy**2]
    ], dtype=np.float64)

    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a normalized quaternion (qw, qx, qy, qz).
    """
    R = np.array(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-8)


def get_camera_heading(q):
    """
    Extract 2D viewing heading vector (dx, dy) in horizontal plane from quaternion.
    """
    R = quaternion_to_rotation_matrix(q)
    # Forward vector assuming camera looks along +Z or +Y axis
    forward = R @ np.array([0.0, 0.0, 1.0])
    return forward[0], forward[1]
