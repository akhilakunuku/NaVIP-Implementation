"""
Evaluation Script for ResNet PoseNet on Stanford 2D-3D-S Dataset.
Calculates mean/median position error, threshold position accuracy (0.25m, 0.50m, 1.00m),
and mean/median orientation error in degrees.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet_posenet import ResNetPoseNet
from training.stanford_dataset import StanfordPoseDataset


def evaluate_stanford(
    checkpoint_path="results/models/navip_stanford_best.pth",
    dataset_dir="dataset/Stanford2D3D",
    csv_file="dataset/Stanford2D3D/processed/test.csv",
    output_dir="results/evaluation"
):
    print("=" * 60)
    print("Evaluating Stanford 2D-3D-S PoseNet Model")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test CSV: {csv_file}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    test_dataset = StanfordPoseDataset(csv_file=csv_file, root_dir=dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load model
    model = ResNetPoseNet(pretrained=False).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Successfully loaded model checkpoint.")
    else:
        print(f"[!] Warning: Checkpoint file not found: {checkpoint_path}. Evaluating with initialized weights.")

    model.eval()

    pos_errors = []
    ori_errors_deg = []

    with torch.no_grad():
        for images, pos_gt, ori_gt in test_loader:
            images = images.to(device)
            pos_gt = pos_gt.numpy()[0]
            ori_gt = ori_gt.numpy()[0]

            pos_pred, ori_pred = model(images)
            pos_pred = pos_pred.cpu().numpy()[0]
            ori_pred = ori_pred.cpu().numpy()[0]

            # 1. Position error (Euclidean distance in meters)
            pos_err = np.linalg.norm(pos_pred - pos_gt)
            pos_errors.append(pos_err)

            # 2. Orientation error (angular difference in degrees)
            dot = np.abs(np.dot(ori_pred, ori_gt))
            dot = np.clip(dot, 0.0, 1.0)
            angle_rad = 2.0 * np.arccos(dot)
            angle_deg = np.degrees(angle_rad)
            ori_errors_deg.append(angle_deg)

    pos_errors = np.array(pos_errors)
    ori_errors_deg = np.array(ori_errors_deg)

    mean_pos_err = float(np.mean(pos_errors))
    median_pos_err = float(np.median(pos_errors))

    acc_025 = float(np.mean(pos_errors <= 0.25) * 100.0)
    acc_050 = float(np.mean(pos_errors <= 0.50) * 100.0)
    acc_100 = float(np.mean(pos_errors <= 1.00) * 100.0)

    mean_ori_err = float(np.mean(ori_errors_deg))
    median_ori_err = float(np.median(ori_errors_deg))

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Test Samples: {len(test_dataset)}")
    print(f"Mean Position Error:     {mean_pos_err:.4f} m")
    print(f"Median Position Error:   {median_pos_err:.4f} m")
    print(f"Position Acc (<= 0.25m): {acc_025:.2f}%")
    print(f"Position Acc (<= 0.50m): {acc_050:.2f}%")
    print(f"Position Acc (<= 1.00m): {acc_100:.2f}%")
    print(f"Mean Orientation Error:   {mean_ori_err:.2f} deg")
    print(f"Median Orientation Error: {median_ori_err:.2f} deg")
    print("=" * 40)

    # Save results summary
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "num_test_samples": len(test_dataset),
        "mean_position_error_m": mean_pos_err,
        "median_position_error_m": median_pos_err,
        "position_accuracy_025m_percent": acc_025,
        "position_accuracy_050m_percent": acc_050,
        "position_accuracy_100m_percent": acc_100,
        "mean_orientation_error_deg": mean_ori_err,
        "median_orientation_error_deg": median_ori_err
    }

    results_json = os.path.join(output_dir, "stanford_evaluation_metrics.json")
    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation metrics to: {results_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stanford PoseNet")
    parser.add_argument("--checkpoint", type=str, default="results/models/navip_stanford_best.pth")
    parser.add_argument("--dataset-dir", type=str, default="dataset/Stanford2D3D")
    parser.add_argument("--csv", type=str, default="dataset/Stanford2D3D/processed/test.csv")
    parser.add_argument("--output-dir", type=str, default="results/evaluation")
    args = parser.parse_args()

    evaluate_stanford(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        csv_file=args.csv,
        output_dir=args.output_dir
    )
