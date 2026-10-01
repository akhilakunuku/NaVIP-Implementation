"""
Evaluation Metrics Module for Pose Estimation.
Computes position errors, orientation angular errors (in degrees),
threshold position accuracy (<=0.25m, <=0.50m, <=1.00m), and summary statistics.
"""

import numpy as np
import torch


def calculate_position_errors(pred_pos, target_pos):
    """
    Calculate Euclidean position errors (meters).
    sqrt((dx)^2 + (dy)^2 + (dz)^2)
    """
    if isinstance(pred_pos, torch.Tensor):
        pred_pos = pred_pos.detach().cpu().numpy()
    if isinstance(target_pos, torch.Tensor):
        target_pos = target_pos.detach().cpu().numpy()

    diff = pred_pos - target_pos
    errors = np.linalg.norm(diff, axis=1)
    return errors


def calculate_orientation_errors_deg(pred_ori, target_ori):
    """
    Calculate orientation angular errors in degrees using normalized quaternions.
    angle = 2 * arccos(|<q1, q2>|) * (180 / pi)
    """
    if isinstance(pred_ori, torch.Tensor):
        pred_ori = pred_ori.detach().cpu().numpy()
    if isinstance(target_ori, torch.Tensor):
        target_ori = target_ori.detach().cpu().numpy()

    # Normalize quaternions to unit length
    pred_norms = np.linalg.norm(pred_ori, axis=1, keepdims=True) + 1e-8
    target_norms = np.linalg.norm(target_ori, axis=1, keepdims=True) + 1e-8
    
    pred_ori_norm = pred_ori / pred_norms
    target_ori_norm = target_ori / target_norms

    # Absolute dot product clamped to [0, 1]
    dot_product = np.abs(np.sum(pred_ori_norm * target_ori_norm, axis=1))
    dot_product = np.clip(dot_product, 0.0, 1.0)

    # Angular error in radians, then convert to degrees
    angle_rad = 2.0 * np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def compute_evaluation_metrics(pred_pos, target_pos, pred_ori, target_ori):
    """
    Computes full set of evaluation metrics.
    Returns dictionary with:
    - mean_position_error_m
    - median_position_error_m
    - position_accuracy_025m_percent
    - position_accuracy_050m_percent
    - position_accuracy_100m_percent
    - mean_orientation_error_deg
    - median_orientation_error_deg
    """
    pos_errors = calculate_position_errors(pred_pos, target_pos)
    ori_errors_deg = calculate_orientation_errors_deg(pred_ori, target_ori)

    mean_pos = float(np.mean(pos_errors))
    median_pos = float(np.median(pos_errors))

    acc_025 = float(np.mean(pos_errors <= 0.25) * 100.0)
    acc_050 = float(np.mean(pos_errors <= 0.50) * 100.0)
    acc_100 = float(np.mean(pos_errors <= 1.00) * 100.0)

    mean_ori = float(np.mean(ori_errors_deg))
    median_ori = float(np.median(ori_errors_deg))

    return {
        "mean_position_error_m": mean_pos,
        "median_position_error_m": median_pos,
        "position_accuracy_025m_percent": acc_025,
        "position_accuracy_050m_percent": acc_050,
        "position_accuracy_100m_percent": acc_100,
        "mean_orientation_error_deg": mean_ori,
        "median_orientation_error_deg": median_ori,
        "raw_pos_errors": pos_errors,
        "raw_ori_errors_deg": ori_errors_deg
    }