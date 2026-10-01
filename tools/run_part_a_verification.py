"""
Part A Verification & Visualization Script for Phase 3.
Generates:
- results/evaluation/training_image_prediction.png
- results/evaluation/validation_image_prediction.png
- results/evaluation/test_image_prediction.png
Verifies:
- results/evaluation/stanford_test_results.csv
- results/evaluation/stanford_test_summary.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet_posenet import ResNetPoseNet
from training.stanford_dataset import StanfordPoseDataset


def create_prediction_visualization(image_raw, img_rel_path, split_name, gt_pos, pred_pos, pos_err, gt_ori, pred_ori, ori_err_deg, save_path):
    """
    Creates a clean 2-panel side-by-side plot:
    Left: RGB input image
    Right: Detailed prediction metrics breakdown card
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.2]})

    # 1. RGB Image Display
    ax1.imshow(image_raw)
    ax1.set_title(f"Input RGB Image ({split_name.capitalize()} Sample)", fontsize=12, fontweight='bold', pad=10)
    ax1.axis("off")

    # 2. Prediction Breakdown Card
    ax2.axis("off")
    
    text_content = (
        f"NaVIP Pose Regression ({split_name.capitalize()} Set)\n"
        f"{'='*45}\n\n"
        f"Image File:\n"
        f"  {os.path.basename(img_rel_path)}\n\n"
        f"POSITION REGRESSION [x, y, z] (meters):\n"
        f"  Ground Truth: [{gt_pos[0]:.4f}, {gt_pos[1]:.4f}, {gt_pos[2]:.4f}]\n"
        f"  Predicted:    [{pred_pos[0]:.4f}, {pred_pos[1]:.4f}, {pred_pos[2]:.4f}]\n"
        f"  Position Error: {pos_err:.4f} meters\n\n"
        f"ORIENTATION REGRESSION [qw, qx, qy, qz]:\n"
        f"  Ground Truth: [{gt_ori[0]:.4f}, {gt_ori[1]:.4f}, {gt_ori[2]:.4f}, {gt_ori[3]:.4f}]\n"
        f"  Predicted:    [{pred_ori[0]:.4f}, {pred_ori[1]:.4f}, {pred_ori[2]:.4f}, {pred_ori[3]:.4f}]\n"
        f"  Orientation Error: {ori_err_deg:.2f} degrees\n"
    )

    ax2.text(
        0.05, 0.95, text_content,
        transform=ax2.transAxes,
        fontsize=10,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f4f6f8', edgecolor='#333333', alpha=0.9)
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}", flush=True)


def run_part_a_verification(
    checkpoint_path="results/models/navip_stanford_best.pth",
    dataset_dir="dataset/Stanford2D3D",
    processed_dir="dataset/Stanford2D3D/processed",
    output_dir="results/evaluation"
):
    print("=" * 60, flush=True)
    print("RUNNING PART A — FINAL PHASE 3 VERIFICATION & VISUALIZATION", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNetPoseNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits_info = [
        ("train", "train.csv", "training_image_prediction.png"),
        ("validation", "validation.csv", "validation_image_prediction.png"),
        ("test", "test.csv", "test_image_prediction.png")
    ]

    single_results = {}

    for split_name, csv_filename, vis_filename in splits_info:
        csv_path = os.path.join(processed_dir, csv_filename)
        ds = StanfordPoseDataset(csv_file=csv_path, root_dir=dataset_dir, preload=False)

        # Index 0 sample
        row = ds.data.iloc[0]
        img_rel_path = row["image"]
        full_img_path = os.path.join(dataset_dir, img_rel_path)
        image_raw = Image.open(full_img_path).convert("RGB")

        image_tensor, gt_pos, gt_ori = ds[0]
        image_batch = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_pos, pred_ori = model(image_batch)

        pred_pos = pred_pos.cpu().numpy()[0]
        pred_ori = pred_ori.cpu().numpy()[0]
        gt_pos = gt_pos.numpy()
        gt_ori = gt_ori.numpy()

        pos_err = float(np.linalg.norm(pred_pos - gt_pos))
        dot = float(np.clip(np.abs(np.dot(pred_ori, gt_ori)), 0.0, 1.0))
        ori_err_deg = float(np.degrees(2.0 * np.arccos(dot)))

        vis_save_path = os.path.join(output_dir, vis_filename)
        create_prediction_visualization(
            image_raw, img_rel_path, split_name,
            gt_pos, pred_pos, pos_err,
            gt_ori, pred_ori, ori_err_deg,
            vis_save_path
        )

        single_results[split_name] = {
            "image": img_rel_path,
            "gt_pos": gt_pos.tolist(),
            "pred_pos": pred_pos.tolist(),
            "pos_err_m": pos_err,
            "gt_ori": gt_ori.tolist(),
            "pred_ori": pred_ori.tolist(),
            "ori_err_deg": ori_err_deg,
            "vis_path": vis_save_path
        }

        print(f"\n[{split_name.upper()} IMAGE RESULT]")
        print(f"  Image path:               {img_rel_path}")
        print(f"  Ground Truth Position:    {gt_pos.round(4).tolist()}")
        print(f"  Predicted Position:       {pred_pos.round(4).tolist()}")
        print(f"  Position Error:           {pos_err:.4f} meters")
        print(f"  Ground Truth Quaternion:  {gt_ori.round(4).tolist()}")
        print(f"  Predicted Quaternion:     {pred_ori.round(4).tolist()}")
        print(f"  Orientation Error:        {ori_err_deg:.2f} degrees")

    # Verify existing Phase 3 evaluation CSV and JSON files
    results_csv = os.path.join(output_dir, "stanford_test_results.csv")
    summary_json = os.path.join(output_dir, "stanford_test_summary.json")

    assert os.path.exists(results_csv), f"Missing CSV: {results_csv}"
    assert os.path.exists(summary_json), f"Missing JSON: {summary_json}"

    df_test_res = pd.read_csv(results_csv)
    with open(summary_json, "r") as f:
        summary_data = json.load(f)

    print(f"\n--- VERIFIED PHASE 3 SUMMARY FILES ---")
    print(f"CSV Rows Count: {len(df_test_res)} (Expected 559)")
    print(f"Summary JSON Num Test Images: {summary_data['num_test_images']}")
    print(f"Mean Position Error:   {summary_data['mean_position_error_m']:.4f} m")
    print(f"Median Position Error: {summary_data['median_position_error_m']:.4f} m")
    print(f"Predictions <= 0.25m:  {summary_data['position_accuracy_025m_percent']:.2f}%")
    print(f"Predictions <= 0.50m:  {summary_data['position_accuracy_050m_percent']:.2f}%")
    print(f"Predictions <= 1.00m:  {summary_data['position_accuracy_100m_percent']:.2f}%")
    print(f"Mean Orientation Error:   {summary_data['mean_orientation_error_deg']:.2f} deg")
    print(f"Median Orientation Error: {summary_data['median_orientation_error_deg']:.2f} deg")

    print("\n" + "=" * 60, flush=True)
    print("PART A VERIFICATION COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 60, flush=True)

    return single_results, summary_data


if __name__ == "__main__":
    run_part_a_verification()
