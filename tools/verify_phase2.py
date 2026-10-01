"""
Phase 2 Lightweight Verification Script for NaVIP.
Verifies all 8 lightweight verification steps (A through H):
A. Load one batch from train.csv (batch_size=4).
B. Run one forward pass on ResNetPoseNet. Verify input [4,3,224,224], position [4,3], orientation [4,4].
C. Verify quaternion norms equal 1.0 (for both predictions and targets).
D. Calculate NaVIPLoss (position loss + orientation loss with sx, sq, beta).
E. Run backward().
F. Verify gradients are non-zero for model weights and learnable uncertainty parameters sx and sq.
G. Run one optimizer step (Adam, lr=1e-4, weight_decay=5e-4).
H. Verify checkpoint save/load works.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet_posenet import ResNetPoseNet
from training.stanford_dataset import StanfordPoseDataset
from training.loss import NaVIPLoss


def run_phase2_verification(
    dataset_dir="dataset/Stanford2D3D",
    processed_dir="dataset/Stanford2D3D/processed",
    test_checkpoint_path="results/models/navip_stanford_verification_test.pth"
):
    print("=" * 60)
    print("STARTING PHASE 2 LIGHTWEIGHT VERIFICATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_csv = os.path.join(processed_dir, "train.csv")

    # Step A: Load one batch from train.csv
    print("\n--- Step A: Load One Batch from train.csv ---")
    dataset = StanfordPoseDataset(csv_file=train_csv, root_dir=dataset_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    images, target_pos, target_ori = next(iter(loader))
    
    images = images.to(device)
    target_pos = target_pos.to(device)
    target_ori = target_ori.to(device)
    
    print(f"Input batch images shape:    {tuple(images.shape)}")
    print(f"Target positions shape:      {tuple(target_pos.shape)}")
    print(f"Target orientations shape:   {tuple(target_ori.shape)}")
    assert images.shape == (4, 3, 224, 224), f"Step A Failed: Image shape is {images.shape}"
    assert target_pos.shape == (4, 3), f"Step A Failed: Target pos shape is {target_pos.shape}"
    assert target_ori.shape == (4, 4), f"Step A Failed: Target ori shape is {target_ori.shape}"
    print("Step A PASSED!")

    # Step B: Run one forward pass
    print("\n--- Step B: Model Forward Pass ---")
    model = ResNetPoseNet(pretrained=True).to(device)
    pred_pos, pred_ori = model(images)

    print(f"Predicted position shape:    {tuple(pred_pos.shape)}")
    print(f"Predicted orientation shape: {tuple(pred_ori.shape)}")
    assert pred_pos.shape == (4, 3), f"Step B Failed: Position shape is {pred_pos.shape}"
    assert pred_ori.shape == (4, 4), f"Step B Failed: Orientation shape is {pred_ori.shape}"
    print("Step B PASSED!")

    # Step C: Verify quaternion norms
    print("\n--- Step C: Quaternion Normalization Verification ---")
    target_norms = torch.norm(target_ori, dim=1)
    pred_norms = torch.norm(pred_ori, dim=1)
    print(f"Target quaternion norms:    {target_norms.detach().cpu().numpy().round(6).tolist()}")
    print(f"Predicted quaternion norms: {pred_norms.detach().cpu().numpy().round(6).tolist()}")
    assert torch.allclose(target_norms, torch.ones_like(target_norms), atol=1e-4), "Target quaternion norms != 1.0"
    assert torch.allclose(pred_norms, torch.ones_like(pred_norms), atol=1e-4), "Predicted quaternion norms != 1.0"
    print("Step C PASSED!")

    # Step D: Calculate loss
    print("\n--- Step D: Loss Calculation ---")
    criterion = NaVIPLoss(initial_sx=0.0, initial_sq=-3.0, beta=1.0).to(device)
    loss, pos_err, ori_err = criterion(pred_pos, pred_ori, target_pos, target_ori)
    print(f"Calculated Total Loss: {loss.item():.6f}")
    print(f"Position Error (pos_err): {pos_err:.6f} m")
    print(f"Orientation Error (ori_err): {ori_err:.6f} rad ({torch.rad2deg(torch.tensor(ori_err)).item():.4f} deg)")
    print(f"sx value: {criterion.sx.item():.6f} | sq value: {criterion.sq.item():.6f}")
    assert torch.isfinite(loss), "Step D Failed: Loss is NaN or Inf"
    print("Step D PASSED!")

    # Step E: Run backward()
    print("\n--- Step E: Run backward() ---")
    optimizer = optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-4,
        weight_decay=5e-4
    )
    optimizer.zero_grad()
    loss.backward()
    print("backward() executed successfully.")
    print("Step E PASSED!")

    # Step F: Verify gradients are non-zero
    print("\n--- Step F: Gradient Verification ---")
    sx_grad = criterion.sx.grad
    sq_grad = criterion.sq.grad
    pos_head_grad = model.position_head.weight.grad
    ori_head_grad = model.orientation_head.weight.grad
    resnet_layer1_grad = model.resnet[4][0].conv1.weight.grad

    print(f"sx gradient: {sx_grad.item() if sx_grad is not None else 'None'}")
    print(f"sq gradient: {sq_grad.item() if sq_grad is not None else 'None'}")
    print(f"Position head weight grad norm: {pos_head_grad.norm().item():.6f}")
    print(f"Orientation head weight grad norm: {ori_head_grad.norm().item():.6f}")
    print(f"ResNet backbone layer1 grad norm: {resnet_layer1_grad.norm().item():.6f}")

    assert sx_grad is not None and abs(sx_grad.item()) > 0, "sx gradient is zero or None!"
    assert sq_grad is not None and abs(sq_grad.item()) > 0, "sq gradient is zero or None!"
    assert pos_head_grad is not None and pos_head_grad.norm().item() > 0, "Position head gradient is zero!"
    assert ori_head_grad is not None and ori_head_grad.norm().item() > 0, "Orientation head gradient is zero!"
    assert resnet_layer1_grad is not None and resnet_layer1_grad.norm().item() > 0, "Backbone gradient is zero!"
    print("Step F PASSED!")

    # Step G: Run one optimizer step
    print("\n--- Step G: Optimizer Step ---")
    initial_pos_weight = model.position_head.weight.clone()
    optimizer.step()
    weight_diff = (model.position_head.weight - initial_pos_weight).norm().item()
    print(f"Position head weight change after step: {weight_diff:.6e}")
    assert weight_diff > 0, "Step G Failed: Weights did not update after optimizer.step()"
    print("Step G PASSED!")

    # Step H: Verify Checkpoint Save/Load
    print("\n--- Step H: Checkpoint Save and Load Verification ---")
    os.makedirs(os.path.dirname(test_checkpoint_path), exist_ok=True)
    checkpoint_dict = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": loss.item(),
        "sx": criterion.sx.item(),
        "sq": criterion.sq.item(),
        "model_config": {
            "backbone": "ResNet-18",
            "pretrained": True,
            "feature_dim": 512,
            "position_head": "512->3",
            "orientation_head": "512->4",
            "input_size": [224, 224]
        }
    }
    torch.save(checkpoint_dict, test_checkpoint_path)
    print(f"Checkpoint saved to: {test_checkpoint_path}")

    # Reload checkpoint
    loaded_ckpt = torch.load(test_checkpoint_path, map_location=device)
    new_model = ResNetPoseNet(pretrained=False).to(device)
    new_model.load_state_dict(loaded_ckpt["model_state_dict"])
    
    # Forward pass on both models in eval mode
    model.eval()
    new_model.eval()
    with torch.no_grad():
        curr_pred_pos, curr_pred_ori = model(images)
        new_pred_pos, new_pred_ori = new_model(images)
        
    pos_diff = (new_pred_pos - curr_pred_pos).abs().max().item()
    ori_diff = (new_pred_ori - curr_pred_ori).abs().max().item()
    print(f"Max difference between original and loaded model outputs: pos_diff={pos_diff:.8f}, ori_diff={ori_diff:.8f}")
    assert pos_diff < 1e-5 and ori_diff < 1e-5, "Step H Failed: Loaded model outputs do not match original model"
    print(f"Loaded epoch: {loaded_ckpt['epoch']}, val_loss: {loaded_ckpt['validation_loss']:.6f}, sx: {loaded_ckpt['sx']:.4f}, sq: {loaded_ckpt['sq']:.4f}")
    print("Step H PASSED!")

    # Clean up test checkpoint file
    if os.path.exists(test_checkpoint_path):
        os.remove(test_checkpoint_path)

    print("\n" + "=" * 60)
    print("ALL PHASE 2 LIGHTWEIGHT VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_phase2_verification()
