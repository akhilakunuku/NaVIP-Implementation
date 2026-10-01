"""
Stanford 2D-3D-Semantics PyTorch Dataset.
Loads images, resizes to 224x224, applies ImageNet normalization,
and returns normalized 3D position [x,y,z] and orientation quaternion [qw,qx,qy,qz].
Includes optional multi-threaded RAM pre-caching for accelerated CPU training.
"""

import os
import concurrent.futures
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class StanfordPoseDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, cache_in_memory=True, preload=True, num_workers=8):
        """
        Args:
            csv_file (str): Path to processed CSV file (train.csv, validation.csv, test.csv).
            root_dir (str): Root directory of the Stanford dataset images.
            transform (callable, optional): PyTorch transform. Default uses ImageNet normalization.
            cache_in_memory (bool): If True, caches preprocessed tensors in memory.
            preload (bool): If True, preloads all images in parallel at initialization.
            num_workers (int): Number of worker threads for preloading images.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = os.path.abspath(root_dir)
        self.cache_in_memory = cache_in_memory
        self.cache = {}

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        if self.cache_in_memory and preload:
            self._preload_cache(num_workers=num_workers)

    def _load_single_sample(self, index):
        row = self.data.iloc[index]
        image_path = os.path.join(self.root_dir, row["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        position = torch.tensor(
            [float(row["x"]), float(row["y"]), float(row["z"])],
            dtype=torch.float32
        )

        orientation = torch.tensor(
            [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
            dtype=torch.float32
        )
        orientation = orientation / (torch.norm(orientation) + 1e-8)

        return index, (image_tensor, position, orientation)

    def _preload_cache(self, num_workers=8):
        print(f"Pre-loading {len(self.data)} samples into RAM cache ({num_workers} threads)...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(self._load_single_sample, range(len(self.data)))
            for idx, sample in results:
                self.cache[idx] = sample
        print(f"Successfully cached {len(self.cache)} samples in RAM.", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache_in_memory and index in self.cache:
            return self.cache[index]

        _, sample = self._load_single_sample(index)
        if self.cache_in_memory:
            self.cache[index] = sample

        return sample
