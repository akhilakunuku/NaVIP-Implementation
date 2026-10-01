import pandas as pd


# ----------------------------------------
# Load results
# ----------------------------------------

FILE = "results/evaluation/test_predictions.csv"

data = pd.read_csv(FILE)


# ----------------------------------------
# Overall statistics
# ----------------------------------------

print()
print("======================================")
print("NaVIP Accuracy Analysis")
print("======================================")

print(f"Test samples: {len(data)}")

print()
print("POSITION ERROR")
print("--------------------------------------")

print(
    f"Average: {data['position_error'].mean():.6f}"
)

print(
    f"Minimum: {data['position_error'].min():.6f}"
)

print(
    f"Maximum: {data['position_error'].max():.6f}"
)


print()
print("ORIENTATION ERROR")
print("--------------------------------------")

print(
    f"Average: {data['orientation_error'].mean():.6f}"
)

print(
    f"Minimum: {data['orientation_error'].min():.6f}"
)

print(
    f"Maximum: {data['orientation_error'].max():.6f}"
)


# ----------------------------------------
# Best predictions
# ----------------------------------------

print()
print("======================================")
print("BEST 5 PREDICTIONS")
print("======================================")

best = data.sort_values(
    "position_error"
).head(5)

print(
    best[
        [
            "image",
            "sequence",
            "position_error",
            "orientation_error"
        ]
    ].to_string(index=False)
)


# ----------------------------------------
# Worst predictions
# ----------------------------------------

print()
print("======================================")
print("WORST 5 PREDICTIONS")
print("======================================")

worst = data.sort_values(
    "position_error",
    ascending=False
).head(5)

print(
    worst[
        [
            "image",
            "sequence",
            "position_error",
            "orientation_error"
        ]
    ].to_string(index=False)
)


# ----------------------------------------
# Sequence-wise analysis
# ----------------------------------------

print()
print("======================================")
print("SEQUENCE-WISE ANALYSIS")
print("======================================")

sequence_results = (
    data
    .groupby("sequence")
    .agg(
        samples=("image", "count"),
        average_position_error=(
            "position_error",
            "mean"
        ),
        average_orientation_error=(
            "orientation_error",
            "mean"
        )
    )
    .reset_index()
)

print(
    sequence_results.to_string(index=False)
)


# ----------------------------------------
# Save analysis
# ----------------------------------------

output_file = (
    "results/evaluation/accuracy_analysis.txt"
)

with open(output_file, "w") as f:

    f.write(
        "NaVIP Accuracy Analysis\n"
    )

    f.write(
        "=======================\n\n"
    )

    f.write(
        f"Test samples: {len(data)}\n\n"
    )

    f.write(
        "POSITION ERROR\n"
    )

    f.write(
        f"Average: "
        f"{data['position_error'].mean():.6f}\n"
    )

    f.write(
        f"Minimum: "
        f"{data['position_error'].min():.6f}\n"
    )

    f.write(
        f"Maximum: "
        f"{data['position_error'].max():.6f}\n\n"
    )

    f.write(
        "ORIENTATION ERROR\n"
    )

    f.write(
        f"Average: "
        f"{data['orientation_error'].mean():.6f}\n"
    )

    f.write(
        f"Minimum: "
        f"{data['orientation_error'].min():.6f}\n"
    )

    f.write(
        f"Maximum: "
        f"{data['orientation_error'].max():.6f}\n\n"
    )

    f.write(
        "SEQUENCE-WISE ANALYSIS\n"
    )

    f.write(
        sequence_results.to_string(index=False)
    )


print()
print("Analysis saved to:")
print(output_file)

print()
print("======================================")
print("Accuracy Analysis Completed")
print("======================================")