import os
import pandas as pd


# ----------------------------------------
# Paths
# ----------------------------------------

INPUT_FILE = (
    "results/evaluation/test_predictions.csv"
)

OUTPUT_DIR = (
    "results/evaluation"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "metrics.txt"
)


# ----------------------------------------
# Load predictions
# ----------------------------------------

print("Loading prediction results...")

data = pd.read_csv(
    INPUT_FILE
)

print(
    "Prediction samples:",
    len(data)
)


# ----------------------------------------
# Position metrics
# ----------------------------------------

position_average = (
    data["position_error"].mean()
)

position_minimum = (
    data["position_error"].min()
)

position_maximum = (
    data["position_error"].max()
)


# ----------------------------------------
# Orientation metrics
# ----------------------------------------

orientation_average = (
    data["orientation_error"].mean()
)

orientation_minimum = (
    data["orientation_error"].min()
)

orientation_maximum = (
    data["orientation_error"].max()
)


# ----------------------------------------
# Print results
# ----------------------------------------

print()
print(
    "======================================"
)

print(
    "NaVIP Pose Evaluation Metrics"
)

print(
    "======================================"
)

print(
    f"Test samples: {len(data)}"
)

print()

print(
    "POSITION ERROR"
)

print(
    f"Average: {position_average:.6f}"
)

print(
    f"Minimum: {position_minimum:.6f}"
)

print(
    f"Maximum: {position_maximum:.6f}"
)

print()

print(
    "ORIENTATION ERROR"
)

print(
    f"Average: {orientation_average:.6f}"
)

print(
    f"Minimum: {orientation_minimum:.6f}"
)

print(
    f"Maximum: {orientation_maximum:.6f}"
)

print(
    "======================================"
)


# ----------------------------------------
# Save metrics
# ----------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w"
) as f:

    f.write(
        "NaVIP Pose Evaluation Metrics\n"
    )

    f.write(
        "==============================\n\n"
    )

    f.write(
        f"Test samples: {len(data)}\n\n"
    )

    f.write(
        "POSITION ERROR\n"
    )

    f.write(
        f"Average: {position_average:.6f}\n"
    )

    f.write(
        f"Minimum: {position_minimum:.6f}\n"
    )

    f.write(
        f"Maximum: {position_maximum:.6f}\n\n"
    )

    f.write(
        "ORIENTATION ERROR\n"
    )

    f.write(
        f"Average: {orientation_average:.6f}\n"
    )

    f.write(
        f"Minimum: {orientation_minimum:.6f}\n"
    )

    f.write(
        f"Maximum: {orientation_maximum:.6f}\n"
    )


print()
print(
    "Metrics saved to:"
)

print(
    OUTPUT_FILE
)