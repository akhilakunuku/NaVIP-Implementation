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
from training.loss import NaVIPLoss


# ----------------------------------------
# Configuration
# ----------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

DATASET_ROOT = "dataset/processed"

TRAIN_CSV = os.path.join(
    DATASET_ROOT,
    "train.csv"
)

VALIDATION_CSV = os.path.join(
    DATASET_ROOT,
    "validation.csv"
)

MODEL_DIR = "results/models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ----------------------------------------
# Dataset
# ----------------------------------------

train_dataset = RealPoseDataset(
    csv_file=TRAIN_CSV,
    dataset_root=DATASET_ROOT,
    image_size=IMAGE_SIZE
)

validation_dataset = RealPoseDataset(
    csv_file=VALIDATION_CSV,
    dataset_root=DATASET_ROOT,
    image_size=IMAGE_SIZE
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ----------------------------------------
# Model
# ----------------------------------------

model = ResNetPoseNet(
    pretrained=True
)

model = model.to(DEVICE)


# ----------------------------------------
# Loss
# ----------------------------------------

criterion = NaVIPLoss()

criterion = criterion.to(DEVICE)


# ----------------------------------------
# Optimizer
# ----------------------------------------

optimizer = torch.optim.Adam(
    list(model.parameters())
    + list(criterion.parameters()),
    lr=LEARNING_RATE
)


# ----------------------------------------
# Training
# ----------------------------------------

print("======================================")
print("NaVIP Real 7-Scenes Training")
print("======================================")

print("Device:", DEVICE)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Epochs:",
    EPOCHS
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for images, true_position, true_orientation in train_loader:

        images = images.to(DEVICE)

        true_position = true_position.to(
            DEVICE
        )

        true_orientation = true_orientation.to(
            DEVICE
        )

        optimizer.zero_grad()

        predicted_position, predicted_orientation = model(
            images
        )

        loss = criterion(
            predicted_position,
            predicted_orientation,
            true_position,
            true_orientation
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_train_loss = (
        total_loss / len(train_loader)
    )


    # ------------------------------------
    # Validation
    # ------------------------------------

    model.eval()

    total_validation_loss = 0.0

    with torch.no_grad():

        for images, true_position, true_orientation in validation_loader:

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

            validation_loss = criterion(
                predicted_position,
                predicted_orientation,
                true_position,
                true_orientation
            )

            total_validation_loss += (
                validation_loss.item()
            )

    average_validation_loss = (
        total_validation_loss
        / len(validation_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Accuracy: {100 * (1 - average_train_loss):.2f}% "
        f"Validation Loss: {average_validation_loss:.4f}"
    )


# ----------------------------------------
# Save model
# ----------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "navip_real_chess.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "loss_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    model_path
)


print("======================================")
print("Training completed.")
print("Model saved to:")
print(model_path)
print("======================================")