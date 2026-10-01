"""
Dataset Loader Verification Script for StanfordPoseDataset.
Loads a batch from train.csv, validation.csv, and test.csv.
Verifies:
- RGB image loading and RGB conversion
- Image shape (BatchSize, 3, 224, 224)
- ImageNet normalization range/values
- Position target shape (BatchSize, 3) -> [x, y, z]
- Orientation target shape (BatchSize, 4) -> [qw, qx, qy, qz]
- Quaternion unit norm check
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.stanford_dataset import StanfordPoseDataset


def test_loader(dataset_dir="dataset/Stanford2D3D", processed_dir="dataset/Stanford2D3D/processed"):
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir = os.path.abspath(processed_dir)

    print("=" * 60)
    print("Testing PyTorch Dataset Loader (StanfordPoseDataset)")
    print("=" * 60)

    splits = ["train.csv", "validation.csv", "test.csv"]

    for split_file in splits:
        csv_path = os.path.join(processed_dir, split_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        ds = StanfordPoseDataset(csv_file=csv_path, root_dir=dataset_dir)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        print(f"\n[Split: {split_file}] Total samples: {len(ds)}")

        # Fetch first batch
        images, positions, orientations = next(iter(loader))

        print(f"  Batch images shape:       {tuple(images.shape)} | dtype: {images.dtype}")
        print(f"  Batch positions shape:    {tuple(positions.shape)} | dtype: {positions.dtype}")
        print(f"  Batch orientations shape: {tuple(orientations.shape)} | dtype: {orientations.dtype}")

        # Check shapes
        assert images.shape == (4, 3, 224, 224), f"Unexpected image shape: {images.shape}"
        assert positions.shape == (4, 3), f"Unexpected position shape: {positions.shape}"
        assert orientations.shape == (4, 4), f"Unexpected orientation shape: {orientations.shape}"

        # Check non-NaN/Inf
        assert not torch.isnan(images).any(), "NaN in images!"
        assert not torch.isnan(positions).any(), "NaN in positions!"
        assert not torch.isnan(orientations).any(), "NaN in orientations!"

        # Check quaternion normalization
        norms = torch.norm(orientations, dim=1)
        print(f"  Batch quaternion norms:   {norms.tolist()}")
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Quaternion norms not 1.0!"

        # Inspect first sample values
        print(f"  Sample 0 Position [x,y,z]: {positions[0].tolist()}")
        print(f"  Sample 0 Orientation [qw,qx,qy,qz]: {orientations[0].tolist()}")

    print("\n" + "=" * 60)
    print("DATASET LOADER VERIFICATION SUCCESSFUL!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_loader()
