import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    ROOT
    / "results"
    / "evaluation"
    / "stanford_test_results.csv"
)


# ============================================================
# TARGET TEST IMAGE
# ============================================================

TARGET_IMAGE = (
    "camera_0e30c45ea0604ddeb7467fd384362503_"
    "office_7_frame_0_domain_rgb.png"
)


# ============================================================
# LOAD RESULTS
# ============================================================

if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"Results file not found:\n{RESULTS_FILE}"
    )

df = pd.read_csv(RESULTS_FILE)

matches = df[
    df["image"].astype(str).str.contains(
        TARGET_IMAGE,
        regex=False
    )
]

if matches.empty:
    raise ValueError(
        f"Test image not found in:\n{RESULTS_FILE}\n\n"
        f"Target:\n{TARGET_IMAGE}"
    )

row = matches.iloc[0]


# ============================================================
# READ VALUES
# ============================================================

image = row["image"]

# Ground truth position
gt_x = float(row["gt_x"])
gt_y = float(row["gt_y"])
gt_z = float(row["gt_z"])

# Predicted position
pred_x = float(row["pred_x"])
pred_y = float(row["pred_y"])
pred_z = float(row["pred_z"])

# Ground truth quaternion
gt_qw = float(row["gt_qw"])
gt_qx = float(row["gt_qx"])
gt_qy = float(row["gt_qy"])
gt_qz = float(row["gt_qz"])

# Predicted quaternion
pred_qw = float(row["pred_qw"])
pred_qx = float(row["pred_qx"])
pred_qy = float(row["pred_qy"])
pred_qz = float(row["pred_qz"])

# Errors
position_error = float(row["position_error_m"])
orientation_error = float(row["orientation_error_deg"])


# ============================================================
# QUATERNION NORM
# ============================================================

pred_quaternion_norm = (
    pred_qw**2
    + pred_qx**2
    + pred_qy**2
    + pred_qz**2
) ** 0.5


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 60)
print("NaVIP — Stanford Pose Prediction")
print("=" * 60)

print()
print("Dataset:")
print("STANFORD 2D-3D-S — AREA 3")

print()
print("Dataset Split:")
print("TEST")

print()
print("Input Image:")
print(image)

print()
print("=" * 60)
print("GROUND TRUTH")
print("=" * 60)

print()
print("Position [x, y, z]:")
print(f"[{gt_x:.4f}, {gt_y:.4f}, {gt_z:.4f}] m")

print()
print("Orientation [qw, qx, qy, qz]:")
print(f"[{gt_qw:.4f}, {gt_qx:.4f}, {gt_qy:.4f}, {gt_qz:.4f}]")

print()
print("=" * 60)
print("PREDICTION")
print("=" * 60)

print()
print("Position [x, y, z]:")
print(f"[{pred_x:.4f}, {pred_y:.4f}, {pred_z:.4f}] m")

print()
print("Orientation [qw, qx, qy, qz]:")
print(
    f"[{pred_qw:.4f}, {pred_qx:.4f}, "
    f"{pred_qy:.4f}, {pred_qz:.4f}]"
)

print()
print("=" * 60)
print("ERROR")
print("=" * 60)

print()
print("Position Prediction Error:")
print(f"{position_error:.4f} m")

print()
print("Orientation Prediction Error:")
print(f"{orientation_error:.4f} degrees")

print()
print("=" * 60)
print("POSE PREDICTION STATUS")
print("=" * 60)

print()
print("Model:")
print("ResNet-18 PoseNet")

print()
print("Checkpoint:")
print("results/models/navip_stanford_best.pth")

print()
print("Quaternion Norm:")
print(f"{pred_quaternion_norm:.6f}")

print()
print("Inference Status:")
print("SUCCESS")

print()
print("=" * 60)
print("=" * 60)
print()