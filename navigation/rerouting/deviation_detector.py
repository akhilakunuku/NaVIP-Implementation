"""
deviation_detector.py - Deviation Detection Module

Calculates distance between current 3D position and active route line segments.
Determines whether user position has deviated beyond configurable distance threshold.
"""

import numpy as np


def compute_perpendicular_distance(point, seg_start, seg_end):
    """
    Computes shortest 3D distance from point to line segment defined by seg_start and seg_end.
    """
    P = np.array(point, dtype=np.float64)
    A = np.array(seg_start, dtype=np.float64)
    B = np.array(seg_end, dtype=np.float64)

    AB = B - A
    length_sq = np.dot(AB, AB)
    if length_sq < 1e-8:
        return np.linalg.norm(P - A)

    # Projection factor t of P onto line AB
    t = np.dot(P - A, AB) / length_sq
    t = np.clip(t, 0.0, 1.0)

    projection = A + t * AB
    return np.linalg.norm(P - projection)


def is_deviated(current_position, seg_start, seg_end, threshold=0.4):
    """
    Evaluates if current_position has strayed beyond threshold meters from segment [seg_start, seg_end].
    """
    dist = compute_perpendicular_distance(current_position, seg_start, seg_end)
    return dist > threshold, dist


if __name__ == "__main__":
    p_on_route = [0.1, 0.02, 0.0]
    p_off_route = [0.1, 0.85, 0.0]
    A = [0.0, 0.0, 0.0]
    B = [1.0, 0.0, 0.0]

    dev_on, d_on = is_deviated(p_on_route, A, B, threshold=0.4)
    dev_off, d_off = is_deviated(p_off_route, A, B, threshold=0.4)

    print("==================================================")
    print("Deviation Detector Distance Test")
    print("==================================================")
    print(f"On-Route Point Dist:  {d_on:.4f}m -> Deviated: {dev_on}")
    print(f"Off-Route Point Dist: {d_off:.4f}m -> Deviated: {dev_off}")
    print("==================================================")
