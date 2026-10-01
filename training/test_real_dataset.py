import os
import sys

from torch.utils.data import DataLoader

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from training.real_dataset import RealPoseDataset


# ----------------------------------------
# Configuration
# ----------------------------------------

DATASET_ROOT = "dataset/processed"
IMAGE_SIZE = 224
BATCH_SIZE = 2


# ----------------------------------------
# Test each split
# ----------------------------------------

splits = [
    "train",
    "validation",
    "test"
]


for split in splits:

    print("\n==============================")
    print(f"Testing {split} dataset")
    print("==============================")

    csv_path = os.path.join(
        DATASET_ROOT,
        f"{split}.csv"
    )

    dataset = RealPoseDataset(
        csv_file=csv_path,
        dataset_root=DATASET_ROOT,
        image_size=IMAGE_SIZE
    )

    print(
        "Dataset samples:",
        len(dataset)
    )

    # Load first sample
    image, position, orientation = dataset[0]

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Position shape:",
        position.shape
    )

    print(
        "Orientation shape:",
        orientation.shape
    )

    # Check actual image path
    row = dataset.data.iloc[0]

    image_path = os.path.join(
        DATASET_ROOT,
        row["image"]
    )

    print(
        "Image exists:",
        os.path.exists(image_path)
    )

    print(
        "Image path:",
        image_path
    )

    # DataLoader test
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    images, positions, orientations = next(
        iter(loader)
    )

    print("Batch images:", images.shape)
    print("Batch positions:", positions.shape)
    print("Batch orientations:", orientations.shape)


print("\n================================")
print("All dataset tests completed")
print("================================")