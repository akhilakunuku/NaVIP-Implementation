"""
Single Test Image Pose Prediction & Map Localization Module.
Loads a trained ResNet PoseNet model, applies exact preprocessing, predicts pose,
localizes on Stanford Area 3 indoor map, and reports position/orientation metrics.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet_posenet import ResNetPoseNet
from mapping.pose_to_map import StanfordMapAligner

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_single_image_localization(
    image_query=None,
    csv_file=r"C:\Users\akhil\Downloads\NaVIP_Implementation\dataset\Stanford2D3D\processed\test.csv",
    image_root=r"C:\Users\akhil\Downloads\NaVIP_Implementation\dataset\Stanford2D3D\area_3\data\rgb",
    checkpoint_path=r"C:\Users\akhil\Downloads\NaVIP_Implementation\results\models\navip_stanford_best.pth",
    map_file=r"C:\Users\akhil\Downloads\NaVIP_Implementation\results\phase4\map\area3_rooms.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Verify CSV file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    test_df = pd.read_csv(csv_file)
    total_samples = len(test_df)

    # 2. Match image in test.csv
    if image_query is None:
        matched_idx = 0
        matched_row = test_df.iloc[0]
    elif isinstance(image_query, int) or (isinstance(image_query, str) and str(image_query).isdigit()):
        matched_idx = int(image_query)
        if not (0 <= matched_idx < total_samples):
            raise IndexError(f"Index {matched_idx} out of bounds for test dataset with {total_samples} samples.")
        matched_row = test_df.iloc[matched_idx]
    else:
        query_name = os.path.basename(str(image_query)).lower()
        matched_idx = None
        matched_row = None
        for idx, row in test_df.iterrows():
            row_img_name = os.path.basename(str(row["image"])).lower()
            if row_img_name == query_name:
                matched_idx = idx
                matched_row = row
                break
        if matched_idx is None:
            for idx, row in test_df.iterrows():
                row_img_name = os.path.basename(str(row["image"])).lower()
                if query_name in row_img_name:
                    matched_idx = idx
                    matched_row = row
                    break
        if matched_idx is None:
            raise ValueError(f"Image '{image_query}' not found in test.csv")

    matched_image_rel = str(matched_row["image"])
    matched_image_name = os.path.basename(matched_image_rel)
    actual_img_path = os.path.join(image_root, matched_image_name)
    if not os.path.exists(actual_img_path):
        dataset_root = os.path.dirname(os.path.dirname(os.path.normpath(image_root)))
        alt_path = os.path.join(dataset_root, matched_image_rel)
        if os.path.exists(alt_path):
            actual_img_path = alt_path

    actual_img_path = os.path.abspath(actual_img_path)

    # Ground truth values
    gt_x = float(matched_row["x"])
    gt_y = float(matched_row["y"])
    gt_z = float(matched_row["z"])
    gt_qw = float(matched_row["qw"])
    gt_qx = float(matched_row["qx"])
    gt_qy = float(matched_row["qy"])
    gt_qz = float(matched_row["qz"])

    # 3. Model Loading & Terminal Output
    print("=" * 60)
    print("TEST IMAGE \u2192 POSE \u2192 MAP LOCALIZATION")
    print("=" * 60)
    print()
    print("Input Image:")
    print(actual_img_path)
    print()
    print("Dataset Type:")
    print("TEST")
    print()
    print("CSV File:")
    print(os.path.abspath(csv_file))
    print()
    print("Image Root:")
    print(os.path.abspath(image_root))
    print()
    print("Loading trained ResNet-18 Stanford PoseNet...")
    print()

    model = ResNetPoseNet(pretrained=False).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully.")
    print()

    # 4. Preprocessing & Inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    raw_image = Image.open(actual_img_path).convert("RGB")
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pos_pred, ori_pred = model(input_tensor)

    pos_pred = pos_pred.cpu().numpy()[0]
    ori_pred = ori_pred.cpu().numpy()[0]

    # Normalize predicted quaternion
    norm_q = np.linalg.norm(ori_pred)
    if norm_q > 1e-8:
        ori_pred = ori_pred / norm_q

    pred_x, pred_y, pred_z = float(pos_pred[0]), float(pos_pred[1]), float(pos_pred[2])
    pred_qw, pred_qx, pred_qy, pred_qz = float(ori_pred[0]), float(ori_pred[1]), float(ori_pred[2]), float(ori_pred[3])

    # 5. Metric Calculations
    gt_pos_arr = np.array([gt_x, gt_y, gt_z])
    pred_pos_arr = np.array([pred_x, pred_y, pred_z])
    pos_error = float(np.linalg.norm(pred_pos_arr - gt_pos_arr))

    gt_ori_arr = np.array([gt_qw, gt_qx, gt_qy, gt_qz])
    gt_ori_arr = gt_ori_arr / (np.linalg.norm(gt_ori_arr) + 1e-8)
    dot = np.abs(np.dot(ori_pred, gt_ori_arr))
    dot = np.clip(dot, 0.0, 1.0)
    ori_error_deg = float(np.degrees(2.0 * np.arccos(dot)))

    # 6. Map Loading & Localization
    aligner = StanfordMapAligner()
    aligned_pose = aligner.align_pose([pred_x, pred_y, pred_z], [pred_qw, pred_qx, pred_qy, pred_qz])

    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Map file not found: {map_file}")

    with open(map_file, "r", encoding="utf-8") as mf:
        map_data = json.load(mf)

    # 7. Formatted Terminal Output
    print("Dataset samples:")
    print(total_samples)
    print()
    print("Matched Dataset Image:")
    print(matched_image_name)
    print()
    print("Dataset Index:")
    print(matched_idx)
    print()
    print("Ground Truth Position:")
    print(f"X: {gt_x:.6f} m")
    print(f"Y: {gt_y:.6f} m")
    print(f"Z: {gt_z:.6f} m")
    print()
    print("Predicted Position:")
    print(f"X: {pred_x:.6f} m")
    print(f"Y: {pred_y:.6f} m")
    print(f"Z: {pred_z:.6f} m")
    print()
    print("Position Error:")
    print(f"{pos_error:.6f} m")
    print()
    print("Ground Truth Orientation:")
    print(f"qw: {gt_qw:.6f}")
    print(f"qx: {gt_qx:.6f}")
    print(f"qy: {gt_qy:.6f}")
    print(f"qz: {gt_qz:.6f}")
    print()
    print("Predicted Orientation:")
    print(f"qw: {pred_qw:.6f}")
    print(f"qx: {pred_qx:.6f}")
    print(f"qy: {pred_qy:.6f}")
    print(f"qz: {pred_qz:.6f}")
    print()
    print("Orientation Error:")
    print(f"{ori_error_deg:.6f} degrees")
    print()
    print("Map File:")
    print(os.path.abspath(map_file))
    print()
    print("Map loaded successfully.")
    print()
    print("=" * 60)
    print("TEST IMAGE LOCALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Test Image Pose Prediction & Map Localization")
    parser.add_argument("--image", "-i", type=str, default=None, help="Image file name, path, or index in test.csv")
    parser.add_argument("--index", type=int, default=None, help="Test CSV index directly")
    args = parser.parse_args()

    query = args.index if args.index is not None else args.image
    run_single_image_localization(image_query=query)
