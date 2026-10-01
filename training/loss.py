"""
NaVIP Uncertainty-Weighted Multi-Task Pose Loss.
Computes position error and quaternion orientation error weighted by learnable
uncertainty parameters sx and sq.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaVIPLoss(nn.Module):

    def __init__(self, initial_sx=0.0, initial_sq=-3.0, beta=1.0):
        super(NaVIPLoss, self).__init__()

        # Learnable log-variance terms for position and orientation
        self.sx = nn.Parameter(torch.tensor(float(initial_sx)))
        self.sq = nn.Parameter(torch.tensor(float(initial_sq)))
        self.beta = beta

    def forward(self, pred_pos, pred_ori, true_pos, true_ori):
        """
        Args:
            pred_pos (Tensor): Predicted position (B, 3)
            pred_ori (Tensor): Predicted quaternion (B, 4)
            true_pos (Tensor): Ground truth position (B, 3)
            true_ori (Tensor): Ground truth quaternion (B, 4)
        """
        # Ensure quaternions are unit normalized
        pred_ori = F.normalize(pred_ori, p=2, dim=1)
        true_ori = F.normalize(true_ori, p=2, dim=1)

        # Position Loss: L1 Euclidean distance per sample
        pos_error = torch.norm(pred_pos - true_pos, dim=1).mean()

        # Orientation Loss: Relative angular distance between quaternions
        # |<q1, q2>| capped at 1.0 - 1e-7 for acos gradient stability
        dot_product = torch.abs(torch.sum(pred_ori * true_ori, dim=1))
        dot_product = torch.clamp(dot_product, 0.0, 1.0 - 1e-7)
        # Angular error in radians
        ori_error = (2.0 * torch.acos(dot_product)).mean()

        # Uncertainty-weighted multi-task loss
        loss = (
            torch.exp(-self.sx) * pos_error + self.sx
            + self.beta * (torch.exp(-self.sq) * ori_error + self.sq)
        )

        return loss, pos_error.item(), ori_error.item()