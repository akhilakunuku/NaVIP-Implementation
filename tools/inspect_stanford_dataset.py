"""
Official Dataset Inspection Script for Stanford 2D-3D-Semantics (2D-3D-S) Dataset.
Inspects official dataset structure under area_X/data/rgb and area_X/data/pose.
"""

import os
import sys
import glob
import argparse
import json


def inspect_stanford_dataset(dataset_dir):
    dataset_dir = os.path.abspath(dataset_dir)
    print("=" * 60)
    print(f"Inspecting Stanford 2D-3D-S Dataset at: {dataset_dir}")
    print("=" * 60)

    if not os.path.exists(dataset_dir):
        return {
            "detected": False,
            "areas": [],
            "rgb_count": 0,
            "pose_count": 0,
            "matched_pairs": 0,
            "unmatched_rgb": 0,
            "unmatched_pose": 0,
            "3d_files": [],
            "rgb_pose_matching": False
        }

    image_extensions = ('.png', '.jpg', '.jpeg')
    official_areas = {'area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6'}

    rgb_images = []
    pose_files = []
    three_d_files = []
    detected_areas = set()

    for root, dirs, files in os.walk(dataset_dir):
        rel_path = os.path.relpath(root, dataset_dir)
        parts = [p.lower() for p in rel_path.split(os.sep)]
        
        for part in parts:
            if part in official_areas or part.startswith('area_'):
                detected_areas.add(part)

        # 3D geometry files under 3d folder
        if os.path.basename(root).lower() == '3d':
            for file in files:
                if file.lower().endswith(('.mat', '.obj', '.ply')):
                    three_d_files.append(os.path.join(root, file))

        # Focus on data/rgb and data/pose directories for perspective frames
        norm_root = root.replace("\\", "/").lower()
        if norm_root.endswith('/data/rgb') or '/data/rgb/' in norm_root:
            for file in files:
                if file.lower().endswith(image_extensions):
                    rgb_images.append(os.path.join(root, file))

        if norm_root.endswith('/data/pose') or '/data/pose/' in norm_root:
            for file in files:
                if file.lower().endswith('.json'):
                    pose_files.append(os.path.join(root, file))

    # Check RGB <-> Pose matching strictly within data/rgb and data/pose
    rgb_bases = {}
    for img_path in rgb_images:
        img_name = os.path.basename(img_path)
        base = img_name.replace('_domain_rgb.png', '').replace('_domain_rgb.jpg', '').replace('_rgb.png', '').replace('_rgb.jpg', '').replace('.png', '').replace('.jpg', '')
        rgb_bases[base] = img_path

    pose_bases = {}
    for pose_path in pose_files:
        pose_name = os.path.basename(pose_path)
        base = pose_name.replace('_domain_pose.json', '').replace('_pose.json', '').replace('.json', '')
        pose_bases[base] = pose_path

    matched_bases = set(rgb_bases.keys()).intersection(set(pose_bases.keys()))
    matched_pairs = len(matched_bases)
    unmatched_rgb = len(rgb_images) - matched_pairs
    unmatched_pose = len(pose_files) - matched_pairs

    matching_passed = (matched_pairs > 0 and unmatched_rgb == 0 and unmatched_pose == 0)
    is_detected = (len(rgb_images) > 0 and len(pose_files) > 0)

    res = {
        "detected": is_detected,
        "areas": sorted(list(detected_areas)),
        "rgb_count": len(rgb_images),
        "pose_count": len(pose_files),
        "matched_pairs": matched_pairs,
        "unmatched_rgb": unmatched_rgb,
        "unmatched_pose": unmatched_pose,
        "3d_files": [os.path.relpath(f, dataset_dir).replace("\\", "/") for f in three_d_files],
        "rgb_pose_matching": matching_passed
    }

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect Real Stanford 2D-3D-S Dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join("dataset", "Stanford2D3D"),
        help="Path to Stanford 2D-3D-S dataset"
    )
    args = parser.parse_args()

    info = inspect_stanford_dataset(args.dataset_dir)
    print(json.dumps(info, indent=2))
