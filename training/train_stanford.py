"""
Training & Reporting Script for ResNet-18 PoseNet on Stanford 2D-3D-S Dataset.
Supports model training and displays the official NaVIP model training report.
"""

import os
import sys
import json
import math
import argparse
import torch
from torch.utils.data import DataLoader

# Reconfigure stdout for UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet_posenet import ResNetPoseNet
from training.stanford_dataset import StanfordPoseDataset
from training.loss import NaVIPLoss


def display_training_report(
    history_file="results/models/training_history.json",
    output_model_dir="results/models",
    train_samples=2585,
    val_samples=560,
    epochs=20,
    batch_size=32,
    lr=0.0001,
    device_name="CPU"
):
    """
    Displays the official NaVIP Model Training Report using actual historical values.
    """
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"Training history file not found: {history_file}")

    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise ValueError("Training history is empty.")

    best_entry = min(history, key=lambda x: x["val_loss"])
    total_epochs = len(history)

    best_model_path = os.path.join(output_model_dir, "navip_stanford_best.pth")
    final_model_path = os.path.join(output_model_dir, "navip_stanford_final.pth")

    print("============================================================")
    print("              NAVIP MODEL TRAINING")
    print("============================================================")
    print()
    print("Dataset:")
    print("Stanford 2D-3D-S — Area 3")
    print()
    print("Training Samples:")
    print(f"{train_samples:,}")
    print()
    print("Validation Samples:")
    print(f"{val_samples:,}")
    print()
    print("Model:")
    print("ResNet-18 PoseNet")
    print()
    print("Input:")
    print("RGB Image")
    print()
    print("Output:")
    print("3D Position (X, Y, Z)")
    print("3D Orientation (qw, qx, qy, qz)")
    print()
    print("Training Configuration:")
    print(f"Epochs: {total_epochs}")
    print(f"Batch Size: {batch_size}")
    print("Optimizer: Adam")
    print(f"Learning Rate: {lr}")
    print(f"Device: {device_name}")
    print()
    print("============================================================")
    print("                 TRAINING PROGRESS")
    print("============================================================")
    print()

    for item in history:
        ep = item["epoch"]
        tr_loss = float(item["train_loss"])
        tr_pos = float(item["train_pos_err"])
        tr_ori_rad = float(item["train_ori_err"])
        tr_ori_deg = tr_ori_rad * (180.0 / math.pi)

        print(f"Epoch {ep}/{total_epochs}")
        print(f"Training Loss: {tr_loss:.4f}")
        print(f"Position Error: {tr_pos:.4f} m")
        print(f"Orientation Error: {tr_ori_deg:.2f}°")
        print()

    best_epoch_num = best_entry["epoch"]
    val_loss = float(best_entry["val_loss"])
    val_pos = float(best_entry["val_pos_err"])
    val_ori_rad = float(best_entry["val_ori_err"])
    val_ori_deg = val_ori_rad * (180.0 / math.pi)

    print("============================================================")
    print("                 VALIDATION RESULT")
    print("============================================================")
    print()
    print("Best Validation Epoch:")
    print(best_epoch_num)
    print()
    print("Validation Loss:")
    print(f"{val_loss:.4f}")
    print()
    print("Validation Position Error:")
    print(f"{val_pos:.4f} m")
    print()
    print("Validation Orientation Error:")
    print(f"{val_ori_deg:.2f}°")
    print()
    print("============================================================")
    print("                 MODEL SAVED")
    print("============================================================")
    print()
    print("Best Model:")
    print(os.path.normpath(best_model_path))
    print()
    print("Final Model:")
    print(os.path.normpath(final_model_path))
    print()
    print("============================================================")
    print("              TRAINING COMPLETED")
    print("============================================================")


def train_stanford(
    dataset_dir="dataset/Stanford2D3D",
    processed_dir="dataset/Stanford2D3D/processed",
    output_model_dir="results/models",
    epochs=20,
    batch_size=32,
    lr=1e-4,
    weight_decay=5e-4,
    pretrained=True,
    dry_run=False
):
    """
    Executes training loop for ResNet-18 PoseNet on Stanford Area 3 dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(processed_dir, "train.csv")
    val_csv = os.path.join(processed_dir, "validation.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(
            f"Processed CSVs not found in {processed_dir}. "
            "Please run tools/prepare_stanford_dataset.py first."
        )

    # Data loaders
    train_dataset = StanfordPoseDataset(csv_file=train_csv, root_dir=dataset_dir, preload=True)
    val_dataset = StanfordPoseDataset(csv_file=val_csv, root_dir=dataset_dir, preload=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNetPoseNet(pretrained=pretrained).to(device)
    criterion = NaVIPLoss().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    os.makedirs(output_model_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    num_epochs = 1 if dry_run else epochs

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_pos_total = 0.0
        train_ori_total = 0.0

        train_count = 0
        for images, pos_gt, ori_gt in train_loader:
            images = images.to(device)
            pos_gt = pos_gt.to(device)
            ori_gt = ori_gt.to(device)

            optimizer.zero_grad()
            pos_pred, ori_pred = model(images)
            loss, pos_err, ori_err = criterion(pos_pred, ori_pred, pos_gt, ori_gt)

            loss.backward()
            optimizer.step()

            batch_count = len(images)
            train_loss_total += loss.item() * batch_count
            train_pos_total += pos_err * batch_count
            train_ori_total += ori_err * batch_count
            train_count += batch_count

            if dry_run:
                break

        train_loss = train_loss_total / train_count
        train_pos = train_pos_total / train_count
        train_ori = train_ori_total / train_count

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_pos_total = 0.0
        val_ori_total = 0.0
        val_count = 0

        with torch.no_grad():
            for images, pos_gt, ori_gt in val_loader:
                images = images.to(device)
                pos_gt = pos_gt.to(device)
                ori_gt = ori_gt.to(device)

                pos_pred, ori_pred = model(images)
                loss, pos_err, ori_err = criterion(pos_pred, ori_pred, pos_gt, ori_gt)

                batch_count = len(images)
                val_loss_total += loss.item() * batch_count
                val_pos_total += pos_err * batch_count
                val_ori_total += ori_err * batch_count
                val_count += batch_count

                if dry_run:
                    break

        val_loss = val_loss_total / val_count
        val_pos = val_pos_total / val_count
        val_ori = val_ori_total / val_count

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pos_err": train_pos,
            "train_ori_err": train_ori,
            "val_loss": val_loss,
            "val_pos_err": val_pos,
            "val_ori_err": val_ori,
            "sx": criterion.sx.item(),
            "sq": criterion.sq.item()
        }
        history.append(history_entry)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_pos_err": train_pos,
            "train_ori_err": train_ori,
            "validation_loss": val_loss,
            "val_pos_err": val_pos,
            "val_ori_err": val_ori,
            "sx": criterion.sx.item(),
            "sq": criterion.sq.item(),
            "model_config": {
                "backbone": "ResNet-18",
                "pretrained": pretrained,
                "input_size": [224, 224]
            }
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(output_model_dir, "navip_stanford_best.pth")
            torch.save(checkpoint_dict, best_path)

        final_path = os.path.join(output_model_dir, "navip_stanford_final.pth")
        torch.save(checkpoint_dict, final_path)

    history_path = os.path.join(output_model_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NaVIP Stanford PoseNet Training & Report")
    parser.add_argument("--dataset-dir", type=str, default="dataset/Stanford2D3D")
    parser.add_argument("--processed-dir", type=str, default="dataset/Stanford2D3D/processed")
    parser.add_argument("--output-model-dir", type=str, default="results/models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train", action="store_true", help="Execute model training loop")
    parser.add_argument("--dry-run", action="store_true", help="Run single step for verification")
    args = parser.parse_args()

    history_path = os.path.join(args.output_model_dir, "training_history.json")

    if args.train:
        train_stanford(
            dataset_dir=args.dataset_dir,
            processed_dir=args.processed_dir,
            output_model_dir=args.output_model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dry_run=args.dry_run
        )

    display_training_report(
        history_file=history_path,
        output_model_dir=args.output_model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
