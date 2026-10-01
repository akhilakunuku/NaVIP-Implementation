import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================
# Project root
# ============================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

# ============================================
# Imports
# ============================================

from models.resnet_posenet import ResNetPoseNet
from training.real_dataset import RealPoseDataset


# ============================================
# Configuration
# ============================================

IMAGE_SIZE = 224
BATCH_SIZE = 8

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed"
)

TEST_CSV = os.path.join(
    DATASET_ROOT,
    "test.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "navip_real_chess.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================
# Load test dataset
# ============================================

print("======================================")
print("NaVIP Base Paper Evaluation")
print("======================================")

print("Device:", DEVICE)

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

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================
# Load model
# ============================================

print()
print("Loading trained ResNet-50 PoseNet...")

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

print("Model loaded successfully.")


# ============================================
# Evaluation storage
# ============================================

position_errors = []
orientation_errors = []


# ============================================
# Evaluation
# ============================================

print()
print("Running test evaluation...")

with torch.no_grad():

    for images, true_position, true_orientation in test_loader:

        images = images.to(DEVICE)

        true_position = (
            true_position
            .to(DEVICE)
        )

        true_orientation = (
            true_orientation
            .to(DEVICE)
        )

        # ------------------------------------
        # Prediction
        # ------------------------------------

        predicted_position, predicted_orientation = model(
            images
        )

        # ------------------------------------
        # Normalize predicted quaternion
        # ------------------------------------

        predicted_orientation = (
            predicted_orientation
            /
            (
                torch.norm(
                    predicted_orientation,
                    dim=1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        # ====================================
        # Position error
        # ====================================

        position_error = torch.norm(
            predicted_position
            - true_position,
            dim=1
        )

        position_errors.extend(
            position_error
            .cpu()
            .numpy()
            .tolist()
        )

        # ====================================
        # Orientation error
        # ====================================

        # Quaternion dot product
        dot_product = torch.sum(
            predicted_orientation
            * true_orientation,
            dim=1
        )

        # q and -q represent the same rotation
        dot_product = torch.abs(
            dot_product
        )

        dot_product = torch.clamp(
            dot_product,
            -1.0,
            1.0
        )

        # Angular error
        angle = (
            2.0
            * torch.acos(
                dot_product
            )
        )

        # Convert radians to degrees
        angle_degrees = (
            angle
            * 180.0
            / np.pi
        )

        orientation_errors.extend(
            angle_degrees
            .cpu()
            .numpy()
            .tolist()
        )


# ============================================
# Calculate statistics
# ============================================

position_errors = np.array(
    position_errors
)

orientation_errors = np.array(
    orientation_errors
)


mean_position_error = (
    np.mean(position_errors)
)

median_position_error = (
    np.median(position_errors)
)

mean_orientation_error = (
    np.mean(orientation_errors)
)

median_orientation_error = (
    np.median(orientation_errors)
)


# ============================================
# Display results
# ============================================

print()
print("======================================")
print("BASE PAPER EVALUATION RESULTS")
print("======================================")

print(
    f"Test samples: {len(position_errors)}"
)

print()

print(
    f"Mean Position Error: "
    f"{mean_position_error:.4f} m"
)

print(
    f"Median Position Error: "
    f"{median_position_error:.4f} m"
)

print()

print(
    f"Mean Orientation Error: "
    f"{mean_orientation_error:.4f} degrees"
)

print(
    f"Median Orientation Error: "
    f"{median_orientation_error:.4f} degrees"
)

print()
print("======================================")
print("Evaluation completed.")
print("======================================")