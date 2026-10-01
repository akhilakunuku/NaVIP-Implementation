import os
import sys

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader


# ============================================
# Project root
# ============================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================
# Imports
# ============================================

from models.resnet_posenet import ResNetPoseNet
from training.real_dataset import RealPoseDataset


# ============================================
# Configuration
# ============================================

IMAGE_SIZE = 224
BATCH_SIZE = 1

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
# Load dataset
# ============================================

dataset = RealPoseDataset(
    csv_file=TEST_CSV,
    dataset_root=DATASET_ROOT,
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================
# Load model
# ============================================

print("Loading trained model...")

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
# Select samples
# ============================================

sample_indices = [
    0,
    50,
    100,
    150,
    199
]


# ============================================
# Prediction
# ============================================

for index in sample_indices:

    image_tensor, true_position, true_orientation = dataset[index]

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    true_position = true_position.numpy()
    true_orientation = true_orientation.numpy()


    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    with torch.no_grad():

        predicted_position, predicted_orientation = model(
            input_tensor
        )


    predicted_position = (
        predicted_position[0]
        .cpu()
        .numpy()
    )

    predicted_orientation = (
        predicted_orientation[0]
        .cpu()
        .numpy()
    )


    # ----------------------------------------
    # Normalize quaternion
    # ----------------------------------------

    predicted_orientation = (
        predicted_orientation
        /
        (
            np.linalg.norm(
                predicted_orientation
            )
            + 1e-8
        )
    )


    # ----------------------------------------
    # Position error
    # ----------------------------------------

    position_error = np.linalg.norm(
        predicted_position
        - true_position
    )


    # ----------------------------------------
    # Orientation error
    # ----------------------------------------

    dot = abs(
        np.dot(
            predicted_orientation,
            true_orientation
        )
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0
    )

    orientation_error = (
        2.0
        * np.arccos(dot)
        * 180.0
        / np.pi
    )


    # ========================================
    # Get original image
    # ========================================

    row = dataset.data.iloc[index]

    image_path = os.path.join(
        DATASET_ROOT,
        row["image"]
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not load:",
            image_path
        )

        continue


    # ========================================
    # Display information
    # ========================================

    print()
    print("======================================")
    print(
        f"Test Sample: {index}"
    )
    print("======================================")

    print(
        "Image:",
        row["image"]
    )

    print()

    print(
        "Ground Truth Position:"
    )

    print(
        f"  x = {true_position[0]:.4f}"
    )

    print(
        f"  y = {true_position[1]:.4f}"
    )

    print(
        f"  z = {true_position[2]:.4f}"
    )

    print()

    print(
        "Predicted Position:"
    )

    print(
        f"  x = {predicted_position[0]:.4f}"
    )

    print(
        f"  y = {predicted_position[1]:.4f}"
    )

    print(
        f"  z = {predicted_position[2]:.4f}"
    )

    print()

    print(
        f"Position Error: "
        f"{position_error:.4f} m"
    )

    print()

    print(
        "Ground Truth Orientation:"
    )

    print(
        f"  qw = {true_orientation[0]:.4f}"
    )

    print(
        f"  qx = {true_orientation[1]:.4f}"
    )

    print(
        f"  qy = {true_orientation[2]:.4f}"
    )

    print(
        f"  qz = {true_orientation[3]:.4f}"
    )

    print()

    print(
        "Predicted Orientation:"
    )

    print(
        f"  qw = {predicted_orientation[0]:.4f}"
    )

    print(
        f"  qx = {predicted_orientation[1]:.4f}"
    )

    print(
        f"  qy = {predicted_orientation[2]:.4f}"
    )

    print(
        f"  qz = {predicted_orientation[3]:.4f}"
    )

    print()

    print(
        f"Orientation Error: "
        f"{orientation_error:.4f} degrees"
    )


    # ========================================
    # Display image
    # ========================================

    display_image = image.copy()

    cv2.putText(
        display_image,
        f"Sample: {index}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_image,
        f"Position Error: {position_error:.3f} m",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_image,
        f"Orientation Error: {orientation_error:.2f} deg",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "NaVIP Pose Prediction",
        display_image
    )

    key = cv2.waitKey(0)

    if key == ord("q"):

        break


cv2.destroyAllWindows()


print()
print("======================================")
print("Prediction visualization completed.")
print("======================================")