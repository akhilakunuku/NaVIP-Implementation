import os
import sys
import json
import math
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.resnet_posenet import ResNetPoseNet
from training.real_dataset import RealPoseDataset


# ============================================================
# CONFIGURATION
# ============================================================

# MODEL_FILE = "results/models/navip_chess_final.pth"
# MAP_FILE = "navigation/map/map_nodes.json"

# DATASET_ROOT = "dataset/processed"
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "navip_chess_final.pth"
)

MAP_FILE = os.path.join(
    PROJECT_ROOT,
    "navigation",
    "map",
    "map_nodes.json"
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# COMMAND LINE
# ============================================================

if len(sys.argv) < 2:

    print()
    print("Usage:")
    print(
        'python .\\navigation\\pose_to_map.py "<image_path>"'
    )

    sys.exit(1)


image_path = os.path.normpath(sys.argv[1])
absolute_image_path = os.path.abspath(image_path)


print("Input Image:")
print(absolute_image_path)


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.isfile(absolute_image_path):

    print()
    print("ERROR: Image file does not exist.")
    print(absolute_image_path)

    sys.exit(1)


# ============================================================
# DETERMINE DATASET TYPE
# ============================================================

normalized_path = absolute_image_path.lower()


if (
    os.sep + "train" + os.sep
    in normalized_path
):

    DATASET_TYPE = "TRAIN"

    CSV_FILE = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "train.csv"
    )

    IMAGE_ROOT = os.path.join(
        DATASET_ROOT,
        "train"
    )


elif (
    os.sep + "validation" + os.sep
    in normalized_path
):

    DATASET_TYPE = "VALIDATION"

    CSV_FILE = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "labels",
        "validation.csv"
    )

    IMAGE_ROOT = os.path.join(
        DATASET_ROOT
    )


elif (
    os.sep + "test" + os.sep
    in normalized_path
):

    DATASET_TYPE = "TEST"

    CSV_FILE = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "labels",
        "test.csv"
    )

    IMAGE_ROOT = os.path.join(
        DATASET_ROOT
    )


else:

    print()
    print(
        "ERROR: Image must be inside train, validation, or test folder."
    )

    sys.exit(1)


CSV_FILE = os.path.abspath(CSV_FILE)
IMAGE_ROOT = os.path.abspath(IMAGE_ROOT)


print()
print("Dataset Type:")
print(DATASET_TYPE)

print()
print("CSV File:")
print(CSV_FILE)

print()
print("Image Root:")
print(IMAGE_ROOT)


# ============================================================
# CHECK CSV
# ============================================================

if not os.path.isfile(CSV_FILE):

    print()
    print("ERROR: CSV file does not exist.")
    print(CSV_FILE)

    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading trained ResNet-50 PoseNet...")


model = ResNetPoseNet(
    pretrained=False
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(DEVICE)

model.eval()


print("Model loaded successfully.")


# ============================================================
# LOAD CSV DIRECTLY
# ============================================================

import pandas as pd


data = pd.read_csv(CSV_FILE)


print()
print("Dataset samples:")
print(len(data))


# ============================================================
# NORMALIZE CSV IMAGE PATH
# ============================================================

def resolve_csv_image(csv_image):

    csv_image = str(csv_image).strip()

    csv_image = csv_image.replace(
        "/",
        os.sep
    )

    csv_image = os.path.normpath(
        csv_image
    )

    # --------------------------------------------------------
    # TRAIN CSV
    #
    # train.csv contains:
    #
    # train/seq-01/seq-01/frame-000001.color.png
    #
    # Actual:
    #
    # dataset/processed/train/seq-01/...
    # --------------------------------------------------------

    if DATASET_TYPE == "TRAIN":

        prefix = (
            "train"
            + os.sep
        )

        if csv_image.lower().startswith(
            prefix.lower()
        ):

            csv_image = csv_image[
                len(prefix):
            ]

        return os.path.normpath(
            os.path.join(
                IMAGE_ROOT,
                csv_image
            )
        )

    # --------------------------------------------------------
    # VALIDATION / TEST CSV
    #
    # labels CSV contains:
    #
    # seq-06/seq-06/frame...
    #
    # Actual:
    #
    # dataset/processed/test/seq-03/...
    # or
    # dataset/processed/validation/seq-06/...
    # --------------------------------------------------------

    if DATASET_TYPE == "TEST":

        return os.path.normpath(
            os.path.join(
                DATASET_ROOT,
                "test",
                csv_image
            )
        )

    if DATASET_TYPE == "VALIDATION":

        return os.path.normpath(
            os.path.join(
                DATASET_ROOT,
                "validation",
                csv_image
            )
        )

    return os.path.normpath(
        os.path.join(
            IMAGE_ROOT,
            csv_image
        )
    )


# ============================================================
# FIND EXACT IMAGE
# ============================================================

target_path = os.path.normpath(
    absolute_image_path
)


selected_index = None


for i in range(len(data)):

    csv_image = data.iloc[i]["image"]

    dataset_image_path = resolve_csv_image(
        csv_image
    )

    if dataset_image_path == target_path:

        selected_index = i

        break


# ============================================================
# FALLBACK: MATCH BY FILENAME
# ============================================================

if selected_index is None:

    target_filename = os.path.basename(
        target_path
    )

    for i in range(len(data)):

        csv_filename = os.path.basename(
            str(
                data.iloc[i]["image"]
            )
        )

        if csv_filename == target_filename:

            selected_index = i

            break


# ============================================================
# IMAGE NOT FOUND
# ============================================================

if selected_index is None:

    print()
    print(
        "ERROR: Image was not found in CSV."
    )

    print()
    print("Requested:")
    print(target_path)

    print()
    print("CSV contains examples:")

    for i in range(
        min(5, len(data))
    ):

        print(
            data.iloc[i]["image"]
        )

    sys.exit(1)


# ============================================================
# GET GROUND TRUTH
# ============================================================

row = data.iloc[
    selected_index
]


matched_image = str(
    row["image"]
)


true_position = torch.tensor(
    [
        float(row["x"]),
        float(row["y"]),
        float(row["z"])
    ],
    dtype=torch.float32
)


true_orientation = torch.tensor(
    [
        float(row["qw"]),
        float(row["qx"]),
        float(row["qy"]),
        float(row["qz"])
    ],
    dtype=torch.float32
)


print()
print("Matched Dataset Image:")
print(matched_image)


print()
print("Resolved Image Path:")
print(
    resolve_csv_image(
        matched_image
    )
)


print()
print("Dataset Index:")
print(selected_index)


# ============================================================
# LOAD IMAGE
# ============================================================

from PIL import Image
from torchvision import transforms


resolved_image_path = resolve_csv_image(
    matched_image
)


if not os.path.isfile(
    resolved_image_path
):

    print()
    print(
        "ERROR: Resolved image does not exist:"
    )

    print(
        resolved_image_path
    )

    sys.exit(1)


image = Image.open(
    resolved_image_path
).convert("RGB")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image = transform(
    image
)


# ============================================================
# POSE PREDICTION
# ============================================================

image = image.unsqueeze(
    0
).to(DEVICE)


with torch.no_grad():

    predicted_position, predicted_orientation = model(
        image
    )


predicted_position = (
    predicted_position[0]
    .cpu()
)


predicted_orientation = (
    predicted_orientation[0]
    .cpu()
)


# ============================================================
# POSITION ERROR
# ============================================================

position_error = torch.sqrt(
    torch.sum(
        (
            predicted_position
            - true_position
        ) ** 2
    )
).item()


# ============================================================
# MAP LOADING
# ============================================================

# nearest_node = None
# nearest_distance = None


# if os.path.isfile(MAP_FILE):

#     try:

#         with open(
#             MAP_FILE,
#             "r"
#         ) as file:

#             map_nodes = json.load(
#                 file
#             )


#         def position_distance(
#             predicted,
#             node
#         ):

#             return math.sqrt(
#                 (predicted[0] - node["position"][0]) ** 2
#                 +
#                 (predicted[1] - node["position"][1]) ** 2
#                 +
#                 (predicted[2] - node["position"][2]) ** 2
#             )


#         predicted_xyz = (
#             predicted_position.tolist()
#         )


#         for node in map_nodes:

#             distance = position_distance(
#                 predicted_xyz,
#                 node
#             )

#             if (
#                 distance
#                 < (
#                     nearest_distance
#                     if nearest_distance is not None
#                     else float("inf")
#                 )
#             ):

#                 nearest_distance = distance

#                 nearest_node = node

#     except Exception as e:

#         print()
#         print(
#             "WARNING: Could not load map."
#         )

#         print(e)

# else:

#     print()
#     print(
#         "WARNING: Map file not found:"
#     )

#     print(
#         os.path.abspath(
#             MAP_FILE
#         )
#     )
# ============================================================
# MAP LOADING
# ============================================================

nearest_node = None
nearest_distance = None
map_nodes = []

print()
print("Map File:")
print(MAP_FILE)

if not os.path.isfile(MAP_FILE):

    print()
    print("ERROR: Map file does not exist:")
    print(MAP_FILE)

else:

    try:

        with open(
            MAP_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            map_nodes = json.load(file)

        if not isinstance(map_nodes, list):

            raise ValueError(
                "map_nodes.json must contain a JSON list."
            )

        print()
        print("Map loaded successfully.")
        print("Map nodes:")
        print(len(map_nodes))

        def position_distance(predicted, node):

            return math.sqrt(
                (predicted[0] - node["position"]["x"]) ** 2
                +
                (predicted[1] - node["position"]["y"]) ** 2
                +
                (predicted[2] - node["position"]["z"]) ** 2
            )

        predicted_xyz = predicted_position.tolist()

        for node in map_nodes:

            distance = position_distance(
                predicted_xyz,
                node
            )

            if (
                nearest_distance is None
                or distance < nearest_distance
            ):

                nearest_distance = distance
                nearest_node = node

        if nearest_node is None:

            print()
            print("WARNING: No valid map nodes found.")

    except Exception as e:

        print()
        print("ERROR: Could not load map.")
        print("Reason:")
        print(
            type(e).__name__,
            "-",
            str(e)
        )

        map_nodes = []


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("======================================")
print("NaVIP Pose → Map Localization")
print("======================================")


print()
print("Dataset:")
print(DATASET_TYPE)


print()
print("Input Image:")
print(
    os.path.basename(
        absolute_image_path
    )
)


print()
print("Ground Truth Position:")
print(
    [
        round(
            v,
            4
        )
        for v in true_position.tolist()
    ]
)


print()
print("Predicted Position:")
print(
    [
        round(
            v,
            4
        )
        for v in predicted_position.tolist()
    ]
)


print()
print("Ground Truth Orientation:")
print(
    [
        round(
            v,
            4
        )
        for v in true_orientation.tolist()
    ]
)


print()
print("Predicted Orientation:")
print(
    [
        round(
            v,
            4
        )
        for v in predicted_orientation.tolist()
    ]
)


print()
print("Position Prediction Error:")
print(
    f"{position_error:.4f} m"
)


print()
print("Nearest Map Node:")


if nearest_node is not None:

    print(
        "Node ID:",
        nearest_node["id"]
    )

    print(
        "Sequence:",
        nearest_node["sequence"]
    )

    print(
        "Image:",
        nearest_node["image"]
    )

    print()
    print("Map Localization Error:")

    print(
        f"{nearest_distance:.4f} m"
    )

else:

    print(
        "Map localization unavailable."
    )


print()
print("======================================")