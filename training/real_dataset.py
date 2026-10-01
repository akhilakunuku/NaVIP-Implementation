import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class RealPoseDataset(Dataset):

    def __init__(
        self,
        csv_file,
        dataset_root,
        image_size=224,
        image_root=None
    ):

        self.data = pd.read_csv(csv_file)

        self.dataset_root = dataset_root

        # Actual location of the images
        self.image_root = (
            image_root
            if image_root is not None
            else dataset_root
        )

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = os.path.join(
            self.image_root,
            row["image"]
        )

        if not os.path.isfile(image_path):

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        position = torch.tensor(
            [
                row["x"],
                row["y"],
                row["z"]
            ],
            dtype=torch.float32
        )

        orientation = torch.tensor(
            [
                row["qw"],
                row["qx"],
                row["qy"],
                row["qz"]
            ],
            dtype=torch.float32
        )

        return (
            image,
            position,
            orientation
        )