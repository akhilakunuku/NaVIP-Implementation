import os
import pandas as pd


# ----------------------------------------
# Paths
# ----------------------------------------

PREDICTIONS_FILE = (
    "results/evaluation/test_predictions.csv"
)

METRICS_FILE = (
    "results/evaluation/metrics.txt"
)

SUMMARY_FILE = (
    "results/evaluation/final_evaluation_summary.txt"
)


# ----------------------------------------
# Load prediction results
# ----------------------------------------

data = pd.read_csv(
    PREDICTIONS_FILE
)


# ----------------------------------------
# Calculate summary
# ----------------------------------------

test_samples = len(data)

average_position_error = (
    data["position_error"].mean()
)

minimum_position_error = (
    data["position_error"].min()
)

maximum_position_error = (
    data["position_error"].max()
)

average_orientation_error = (
    data["orientation_error"].mean()
)

minimum_orientation_error = (
    data["orientation_error"].min()
)

maximum_orientation_error = (
    data["orientation_error"].max()
)


# ----------------------------------------
# Save final summary
# ----------------------------------------

os.makedirs(
    "results/evaluation",
    exist_ok=True
)

with open(
    SUMMARY_FILE,
    "w"
) as f:

    f.write(
        "NaVIP REAL DATASET EVALUATION\n"
    )

    f.write(
        "==============================\n\n"
    )

    f.write(
        "Dataset: 7-Scenes Chess\n"
    )

    f.write(
        f"Test samples: {test_samples}\n\n"
    )

    f.write(
        "POSITION ERROR\n"
    )

    f.write(
        f"Average: "
        f"{average_position_error:.6f}\n"
    )

    f.write(
        f"Minimum: "
        f"{minimum_position_error:.6f}\n"
    )

    f.write(
        f"Maximum: "
        f"{maximum_position_error:.6f}\n\n"
    )

    f.write(
        "ORIENTATION ERROR\n"
    )

    f.write(
        f"Average: "
        f"{average_orientation_error:.6f}\n"
    )

    f.write(
        f"Minimum: "
        f"{minimum_orientation_error:.6f}\n"
    )

    f.write(
        f"Maximum: "
        f"{maximum_orientation_error:.6f}\n\n"
    )

    f.write(
        "FILES\n"
    )

    f.write(
        "- test_predictions.csv\n"
    )

    f.write(
        "- metrics.txt\n"
    )


# ----------------------------------------
# Display result
# ----------------------------------------

print(
    "======================================"
)

print(
    "Final Evaluation Summary Created"
)

print(
    "======================================"
)

print(
    f"Test samples: {test_samples}"
)

print(
    f"Average position error: "
    f"{average_position_error:.6f}"
)

print(
    f"Average orientation error: "
    f"{average_orientation_error:.6f}"
)

print()
print(
    "Saved to:"
)

print(
    SUMMARY_FILE
)

print(
    "======================================"
)