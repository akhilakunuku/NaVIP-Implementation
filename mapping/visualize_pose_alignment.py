"""
Diagnostic Script to Visualize Ground Truth vs Predicted Camera Poses
in the Stanford 3D Map Coordinate System.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mapping.stanford_geometry import inspect_ply_geometry
from mapping.pose_to_map import StanfordMapAligner


def visualize_pose_comparison(
    gt_pos, gt_q,
    pred_pos, pred_q,
    scene_name="Stanford Scene",
    geometry_file=None
):
    print("=" * 60)
    print(f"3D Map Alignment Diagnostic: {scene_name}")
    print("=" * 60)

    bounds = None
    if geometry_file and os.path.exists(geometry_file) and geometry_file.endswith('.ply'):
        geom_info = inspect_ply_geometry(geometry_file)
        bounds = geom_info["bounds"]
        print(f"Scene Geometry Bounds from {os.path.basename(geometry_file)}:")
        if bounds:
            print(f"  X: [{bounds['x_min']:.2f}, {bounds['x_max']:.2f}] m")
            print(f"  Y: [{bounds['y_min']:.2f}, {bounds['y_max']:.2f}] m")
            print(f"  Z: [{bounds['z_min']:.2f}, {bounds['z_max']:.2f}] m")

    aligner = StanfordMapAligner(scene_bounds=bounds)

    gt_aligned = aligner.align_pose(gt_pos, gt_q)
    pred_aligned = aligner.align_pose(pred_pos, pred_q)

    pos_diff = np.array(pred_aligned["map_position"]) - np.array(gt_aligned["map_position"])
    pos_err = np.linalg.norm(pos_diff)

    dot = np.abs(np.dot(pred_aligned["quaternion"], gt_aligned["quaternion"]))
    dot = np.clip(dot, 0.0, 1.0)
    ori_err_deg = np.degrees(2.0 * np.arccos(dot))

    print("\n------------------------------------------------------------")
    print("GROUND TRUTH CAMERA POSE:")
    print(f"  Map Position (x,y,z): [{gt_aligned['map_position'][0]:.3f}, {gt_aligned['map_position'][1]:.3f}, {gt_aligned['map_position'][2]:.3f}] meters")
    print(f"  Heading Vector (dx,dy): [{gt_aligned['heading_vector_2d'][0]:.3f}, {gt_aligned['heading_vector_2d'][1]:.3f}]")
    print(f"  Inside Scene Bounds:   {gt_aligned['is_valid_map_coordinate']}")

    print("\nPREDICTED CAMERA POSE:")
    print(f"  Map Position (x,y,z): [{pred_aligned['map_position'][0]:.3f}, {pred_aligned['map_position'][1]:.3f}, {pred_aligned['map_position'][2]:.3f}] meters")
    print(f"  Heading Vector (dx,dy): [{pred_aligned['heading_vector_2d'][0]:.3f}, {pred_aligned['heading_vector_2d'][1]:.3f}]")
    print(f"  Inside Scene Bounds:   {pred_aligned['is_valid_map_coordinate']}")

    print("\nALIGNMENT METRICS:")
    print(f"  Delta X: {pos_diff[0]:+.4f} m | Delta Y: {pos_diff[1]:+.4f} m | Delta Z: {pos_diff[2]:+.4f} m")
    print(f"  Position Error:    {pos_err:.4f} meters")
    print(f"  Orientation Error: {ori_err_deg:.2f} degrees")
    print("=" * 60)


if __name__ == "__main__":
    gt_position = [1.20, 2.30, 1.50]
    gt_quaternion = [1.0, 0.0, 0.0, 0.0]

    pred_position = [1.25, 2.34, 1.48]
    pred_quaternion = [0.998, 0.01, 0.02, 0.03]

    visualize_pose_comparison(gt_position, gt_quaternion, pred_position, pred_quaternion)
