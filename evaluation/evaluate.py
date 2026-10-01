import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.resnet_posenet import ResNetPoseNet
from training.real_dataset import RealPoseDataset
from evaluation.metrics import (
    position_error,
    orientation_error
)


# ----------------------------------------
# Configuration
# ----------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

DATASET_ROOT = "dataset/processed"

TEST_CSV = os.path.join(
    DATASET_ROOT,
    "test.csv"
)

MODEL_PATH = (
    "results/models/navip_real_chess.pth"
)


# ----------------------------------------
# Dataset
# ----------------------------------------

test_dataset = RealPoseDataset(
    csv_file=TEST_CSV,
    dataset_root=DATASET_ROOT,
    image_size=IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ----------------------------------------
# Model
# ----------------------------------------

model = ResNetPoseNet(
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ----------------------------------------
# Evaluation
# ----------------------------------------

total_position_error = 0.0
total_orientation_error = 0.0

total_samples = 0


with torch.no_grad():

    for (
        images,
        true_position,
        true_orientation
    ) in test_loader:

        images = images.to(DEVICE)

        true_position = true_position.to(
            DEVICE
        )

        true_orientation = true_orientation.to(
            DEVICE
        )

        predicted_position, predicted_orientation = model(
            images
        )

        predicted_orientation = (
            predicted_orientation /
            (
                torch.norm(
                    predicted_orientation,
                    dim=1,
                    keepdim=True
                ) + 1e-8
            )
        )

        position_errors = torch.norm(
            predicted_position - true_position,
            dim=1
        )

        orientation_errors = torch.norm(
            predicted_orientation - true_orientation,
            dim=1
        )

        total_position_error += (
            position_errors.sum().item()
        )

        total_orientation_error += (
            orientation_errors.sum().item()
        )

        total_samples += images.size(0)


# ----------------------------------------
# Average errors
# ----------------------------------------

average_position_error = (
    total_position_error
    / total_samples
)

average_orientation_error = (
    total_orientation_error
    / total_samples
)


# ----------------------------------------
# Results
# ----------------------------------------

print("======================================")
print("NaVIP Local Test Evaluation")
print("======================================")

print(
    "Test samples:",
    total_samples
)

print(
    "Average position error:",
    average_position_error
)

print(
    "Average orientation error:",
    average_orientation_error
)

print("======================================")