import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class RealPoseDataset(Dataset):

    def __init__(self, csv_file, root_dir):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = os.path.join(
            self.root_dir,
            row["image"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        position = torch.tensor(
            [
                float(row["x"]),
                float(row["y"]),
                float(row["z"])
            ],
            dtype=torch.float32
        )

        orientation = torch.tensor(
            [
                float(row["qw"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"])
            ],
            dtype=torch.float32
        )

        orientation = orientation / (
            torch.norm(orientation) + 1e-8
        )

        return (
            image,
            position,
            orientation
        )