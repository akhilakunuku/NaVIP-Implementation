"""
Phase 3 — Pose Prediction and Evaluation Script for NaVIP (Stanford 2D-3D-S Area 3).
Performs:
- Step 1: Verification of best checkpoint (results/models/navip_stanford_best.pth)
- Step 2: Prediction on 1 Training image (first image from train.csv)
- Step 3: Prediction on 1 Validation image (first image from validation.csv)
- Step 4: Prediction on 1 Unseen Test image (first image from test.csv)
- Step 5 & 6: Full 559-image Test Set Evaluation and Metric Calculation
- Step 7: Saving results to CSV (results/evaluation/stanford_test_results.csv)
          and summary JSON (results/evaluation/stanford_test_summary.json)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet_posenet import ResNetPoseNet
from training.stanford_dataset import StanfordPoseDataset


def evaluate_single_sample(model, dataset, index, device):
    """
    Evaluates a single sample from the dataset.
    Returns dictionary with image path, GT pose, predicted pose, position error, orientation error.
    """
    row = dataset.data.iloc[index]
    image_rel_path = row["image"]

    image_tensor, gt_pos, gt_ori = dataset[index]
    image_tensor_batch = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_pos, pred_ori = model(image_tensor_batch)

    pred_pos = pred_pos.cpu().numpy()[0]
    pred_ori = pred_ori.cpu().numpy()[0]
    gt_pos = gt_pos.numpy()
    gt_ori = gt_ori.numpy()

    # Position Error: Euclidean distance in meters
    pos_err = np.linalg.norm(pred_pos - gt_pos)

    # Orientation Error: Geodesic angular distance in degrees
    dot = np.abs(np.dot(pred_ori, gt_ori))
    dot = np.clip(dot, 0.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    return {
        "image": image_rel_path,
        "gt_pos": gt_pos,
        "pred_pos": pred_pos,
        "pos_err": pos_err,
        "gt_ori": gt_ori,
        "pred_ori": pred_ori,
        "ori_err_deg": angle_deg
    }


def run_phase3_evaluation(
    checkpoint_path="results/models/navip_stanford_best.pth",
    dataset_dir="dataset/Stanford2D3D",
    processed_dir="dataset/Stanford2D3D/processed",
    output_dir="results/evaluation"
):
    print("=" * 60, flush=True)
    print("STARTING PHASE 3 — POSE PREDICTION & EVALUATION", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ----------------------------------------------------
    # STEP 1: Verify Best Checkpoint
    # ----------------------------------------------------
    print("\n--- STEP 1: VERIFY BEST CHECKPOINT ---", flush=True)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from: {checkpoint_path}", flush=True)

    model = ResNetPoseNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Successfully instantiated ResNetPoseNet and loaded state dict.", flush=True)

    # Test dummy forward pass
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        pos_test, ori_test = model(dummy_input)
    ori_test_norms = torch.norm(ori_test, dim=1)
    assert pos_test.shape == (2, 3), "Dummy forward pass position shape mismatch!"
    assert ori_test.shape == (2, 4), "Dummy forward pass orientation shape mismatch!"
    assert torch.allclose(ori_test_norms, torch.ones_like(ori_test_norms), atol=1e-4), "Quaternion normalization failed!"
    print("Step 1 PASSED: Checkpoint loaded, forward pass works, quaternion outputs normalized.", flush=True)

    # Datasets
    train_csv = os.path.join(processed_dir, "train.csv")
    val_csv = os.path.join(processed_dir, "validation.csv")
    test_csv = os.path.join(processed_dir, "test.csv")

    train_ds = StanfordPoseDataset(csv_file=train_csv, root_dir=dataset_dir, preload=False)
    val_ds = StanfordPoseDataset(csv_file=val_csv, root_dir=dataset_dir, preload=False)
    test_ds = StanfordPoseDataset(csv_file=test_csv, root_dir=dataset_dir, preload=False)

    # ----------------------------------------------------
    # STEP 2: One Training Image
    # ----------------------------------------------------
    print("\n--- STEP 2: ONE TRAINING IMAGE ---", flush=True)
    train_sample = evaluate_single_sample(model, train_ds, 0, device)
    print(f"Image Path: {train_sample['image']}")
    print(f"Ground Truth Position [x, y, z]: {train_sample['gt_pos'].round(4).tolist()}")
    print(f"Predicted Position [x, y, z]:    {train_sample['pred_pos'].round(4).tolist()}")
    print(f"Position Error:                 {train_sample['pos_err']:.4f} meters")
    print(f"Ground Truth Quaternion [qw, qx, qy, qz]: {train_sample['gt_ori'].round(4).tolist()}")
    print(f"Predicted Quaternion [qw, qx, qy, qz]:    {train_sample['pred_ori'].round(4).tolist()}")
    print(f"Orientation Error:              {train_sample['ori_err_deg']:.2f} degrees")

    # ----------------------------------------------------
    # STEP 3: One Validation Image
    # ----------------------------------------------------
    print("\n--- STEP 3: ONE VALIDATION IMAGE ---", flush=True)
    val_sample = evaluate_single_sample(model, val_ds, 0, device)
    print(f"Image Path: {val_sample['image']}")
    print(f"Ground Truth Position [x, y, z]: {val_sample['gt_pos'].round(4).tolist()}")
    print(f"Predicted Position [x, y, z]:    {val_sample['pred_pos'].round(4).tolist()}")
    print(f"Position Error:                 {val_sample['pos_err']:.4f} meters")
    print(f"Ground Truth Quaternion [qw, qx, qy, qz]: {val_sample['gt_ori'].round(4).tolist()}")
    print(f"Predicted Quaternion [qw, qx, qy, qz]:    {val_sample['pred_ori'].round(4).tolist()}")
    print(f"Orientation Error:              {val_sample['ori_err_deg']:.2f} degrees")

    # ----------------------------------------------------
    # STEP 4: One Unseen Test Image
    # ----------------------------------------------------
    print("\n--- STEP 4: ONE UNSEEN TEST IMAGE ---", flush=True)
    test_sample = evaluate_single_sample(model, test_ds, 0, device)
    print(f"Image Path: {test_sample['image']}")
    print(f"Ground Truth Position [x, y, z]: {test_sample['gt_pos'].round(4).tolist()}")
    print(f"Predicted Position [x, y, z]:    {test_sample['pred_pos'].round(4).tolist()}")
    print(f"Position Error:                 {test_sample['pos_err']:.4f} meters")
    print(f"Ground Truth Quaternion [qw, qx, qy, qz]: {test_sample['gt_ori'].round(4).tolist()}")
    print(f"Predicted Quaternion [qw, qx, qy, qz]:    {test_sample['pred_ori'].round(4).tolist()}")
    print(f"Orientation Error:              {test_sample['ori_err_deg']:.2f} degrees")

    # ----------------------------------------------------
    # STEP 5 & 6: Full 559 Test Set Evaluation
    # ----------------------------------------------------
    print("\n--- STEP 5 & 6: FULL TEST SET EVALUATION (559 IMAGES) ---", flush=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    rows_results = []
    pos_errors = []
    ori_errors_deg = []
    successful_predictions = 0
    failed_predictions = 0

    idx_offset = 0
    with torch.no_grad():
        for images, gt_pos_batch, gt_ori_batch in test_loader:
            images = images.to(device)
            pred_pos_batch, pred_ori_batch = model(images)

            pred_pos_batch = pred_pos_batch.cpu().numpy()
            pred_ori_batch = pred_ori_batch.cpu().numpy()
            gt_pos_batch = gt_pos_batch.numpy()
            gt_ori_batch = gt_ori_batch.numpy()

            for i in range(len(images)):
                img_rel_path = test_ds.data.iloc[idx_offset + i]["image"]
                gt_pos = gt_pos_batch[i]
                pred_pos = pred_pos_batch[i]
                gt_ori = gt_ori_batch[i]
                pred_ori = pred_ori_batch[i]

                # Position Error
                p_err = np.linalg.norm(pred_pos - gt_pos)

                # Orientation Error in degrees
                dot = np.abs(np.dot(pred_ori, gt_ori))
                dot = np.clip(dot, 0.0, 1.0)
                o_err_deg = np.degrees(2.0 * np.arccos(dot))

                if np.isfinite(p_err) and np.isfinite(o_err_deg):
                    successful_predictions += 1
                else:
                    failed_predictions += 1

                pos_errors.append(p_err)
                ori_errors_deg.append(o_err_deg)

                rows_results.append({
                    "image": img_rel_path,
                    "gt_x": float(gt_pos[0]),
                    "gt_y": float(gt_pos[1]),
                    "gt_z": float(gt_pos[2]),
                    "pred_x": float(pred_pos[0]),
                    "pred_y": float(pred_pos[1]),
                    "pred_z": float(pred_pos[2]),
                    "position_error_m": float(p_err),
                    "gt_qw": float(gt_ori[0]),
                    "gt_qx": float(gt_ori[1]),
                    "gt_qy": float(gt_ori[2]),
                    "gt_qz": float(gt_ori[3]),
                    "pred_qw": float(pred_ori[0]),
                    "pred_qx": float(pred_ori[1]),
                    "pred_qy": float(pred_ori[2]),
                    "pred_qz": float(pred_ori[3]),
                    "orientation_error_deg": float(o_err_deg)
                })

            idx_offset += len(images)

    pos_errors = np.array(pos_errors)
    ori_errors_deg = np.array(ori_errors_deg)

    mean_pos_err = float(np.mean(pos_errors))
    median_pos_err = float(np.median(pos_errors))
    acc_025 = float(np.mean(pos_errors <= 0.25) * 100.0)
    acc_050 = float(np.mean(pos_errors <= 0.50) * 100.0)
    acc_100 = float(np.mean(pos_errors <= 1.00) * 100.0)

    mean_ori_err = float(np.mean(ori_errors_deg))
    median_ori_err = float(np.median(ori_errors_deg))

    print(f"Total Test Images:       {len(test_ds)}")
    print(f"Successful Predictions:   {successful_predictions}")
    print(f"Failed Predictions:       {failed_predictions}")
    print(f"Mean Position Error:     {mean_pos_err:.4f} meters")
    print(f"Median Position Error:   {median_pos_err:.4f} meters")
    print(f"Position Acc (<= 0.25m): {acc_025:.2f} %")
    print(f"Position Acc (<= 0.50m): {acc_050:.2f} %")
    print(f"Position Acc (<= 1.00m): {acc_100:.2f} %")
    print(f"Mean Orientation Error:   {mean_ori_err:.2f} degrees")
    print(f"Median Orientation Error: {median_ori_err:.2f} degrees")

    # ----------------------------------------------------
    # STEP 7: Save Results (CSV & JSON)
    # ----------------------------------------------------
    print("\n--- STEP 7: SAVE RESULTS ---", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "stanford_test_results.csv")
    df_results = pd.DataFrame(rows_results)
    df_results.to_csv(csv_path, index=False)
    print(f"Saved test prediction details to CSV: {csv_path}", flush=True)

    json_path = os.path.join(output_dir, "stanford_test_summary.json")
    summary = {
        "dataset": "Stanford 2D-3D-S Area 3 Test Set",
        "num_test_images": len(test_ds),
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "mean_position_error_m": mean_pos_err,
        "median_position_error_m": median_pos_err,
        "position_accuracy_025m_percent": acc_025,
        "position_accuracy_050m_percent": acc_050,
        "position_accuracy_100m_percent": acc_100,
        "mean_orientation_error_deg": mean_ori_err,
        "median_orientation_error_deg": median_ori_err,
        "best_checkpoint_evaluated": checkpoint_path
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary JSON to: {json_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("PHASE 3 EVALUATION COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 60, flush=True)

    return {
        "train_sample": train_sample,
        "val_sample": val_sample,
        "test_sample": test_sample,
        "summary": summary,
        "csv_path": csv_path,
        "json_path": json_path
    }


if __name__ == "__main__":
    run_phase3_evaluation()
