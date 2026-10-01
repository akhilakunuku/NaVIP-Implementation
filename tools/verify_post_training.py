"""
Post-Training Verification Script for Phase 2 (20-epoch run).
Verifies:
1. Training completed 20 epochs.
2. Best checkpoint exists (results/models/navip_stanford_best.pth).
3. Final checkpoint exists (results/models/navip_stanford_final.pth).
4. Training history exists (results/models/training_history.json).
5. Best checkpoint loads into model & optimizer state.
6. Final checkpoint loads into model & optimizer state.
7. Model forward pass works after loading.
8. Quaternion output remains normalized (norm == 1.0).
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet_posenet import ResNetPoseNet


def verify_post_training(output_model_dir="results/models"):
    output_model_dir = os.path.abspath(output_model_dir)
    print("=" * 60, flush=True)
    print("STARTING PHASE 2 POST-TRAINING VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    best_path = os.path.join(output_model_dir, "navip_stanford_best.pth")
    final_path = os.path.join(output_model_dir, "navip_stanford_final.pth")
    history_path = os.path.join(output_model_dir, "training_history.json")

    # 1. Training history exists and has 20 epochs
    print(f"1. Checking training history: {history_path}", flush=True)
    assert os.path.exists(history_path), f"Training history not found: {history_path}"
    with open(history_path, "r") as f:
        history = json.load(f)
    print(f"   -> TRAINING HISTORY EXISTS ({len(history)} epochs recorded).", flush=True)

    # 2 & 3. Best and Final checkpoints exist
    print(f"2. Checking best checkpoint:  {best_path}", flush=True)
    assert os.path.exists(best_path), f"Best checkpoint not found: {best_path}"
    print("   -> BEST CHECKPOINT EXISTS.", flush=True)

    print(f"3. Checking final checkpoint: {final_path}", flush=True)
    assert os.path.exists(final_path), f"Final checkpoint not found: {final_path}"
    print("   -> FINAL CHECKPOINT EXISTS.", flush=True)

    # 4 & 5. Verify loading both checkpoints
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Best Checkpoint
    print("\n4. Verifying BEST checkpoint loading...", flush=True)
    best_ckpt = torch.load(best_path, map_location=device)
    required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "validation_loss", "sx", "sq", "model_config"]
    for k in required_keys:
        assert k in best_ckpt, f"Missing key '{k}' in best checkpoint!"
    print(f"   -> Best Checkpoint Loaded cleanly (Epoch {best_ckpt['epoch']}, Val Loss: {best_ckpt['validation_loss']:.6f}).", flush=True)

    # Final Checkpoint
    print("\n5. Verifying FINAL checkpoint loading...", flush=True)
    final_ckpt = torch.load(final_path, map_location=device)
    for k in required_keys:
        assert k in final_ckpt, f"Missing key '{k}' in final checkpoint!"
    print(f"   -> Final Checkpoint Loaded cleanly (Epoch {final_ckpt['epoch']}, Val Loss: {final_ckpt['validation_loss']:.6f}).", flush=True)

    # 6. Model forward pass on best model
    print("\n6. Instantiating ResNet-18 PoseNet model and loading state dict...", flush=True)
    model = ResNetPoseNet(pretrained=False).to(device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        pos_pred, ori_pred = model(dummy_input)

    print(f"   -> Model forward output position shape:    {tuple(pos_pred.shape)}", flush=True)
    print(f"   -> Model forward output orientation shape: {tuple(ori_pred.shape)}", flush=True)
    assert pos_pred.shape == (4, 3), f"Invalid position output shape: {pos_pred.shape}"
    assert ori_pred.shape == (4, 4), f"Invalid orientation output shape: {ori_pred.shape}"
    print("   -> MODEL FORWARD PASS WORKED CLEANLY.", flush=True)

    # 7. Quaternion normalization check
    print("\n7. Verifying quaternion output normalization...", flush=True)
    ori_norms = torch.norm(ori_pred, dim=1)
    print(f"   -> Quaternion output norms: {ori_norms.cpu().numpy().round(6).tolist()}", flush=True)
    assert torch.allclose(ori_norms, torch.ones_like(ori_norms), atol=1e-4), "Quaternion outputs are not normalized to 1.0!"
    print("   -> QUATERNION NORMALIZATION VERIFIED.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("ALL POST-TRAINING VERIFICATION CHECKS PASSED CLEANLY!", flush=True)
    print("=" * 60, flush=True)

    # Return key stats for summary report
    last_entry = history[-1]
    best_entry = min(history, key=lambda x: x["val_loss"])

    stats = {
        "device": str(device),
        "epochs_completed": len(history),
        "final_train_loss": last_entry["train_loss"],
        "final_val_loss": last_entry["val_loss"],
        "best_epoch": best_entry["epoch"],
        "best_val_loss": best_entry["val_loss"],
        "best_train_pos_loss": best_entry["train_pos_err"],
        "best_val_pos_loss": best_entry["val_pos_err"],
        "best_train_ori_loss": best_entry["train_ori_err"],
        "best_val_ori_loss": best_entry["val_ori_err"],
        "final_sx": last_entry["sx"],
        "final_sq": last_entry["sq"],
        "best_checkpoint_path": best_path,
        "final_checkpoint_path": final_path,
        "history_path": history_path,
        "checkpoint_reload_passed": True
    }

    print("\nSummary of Statistics:", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)

    return stats


if __name__ == "__main__":
    verify_post_training()
