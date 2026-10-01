import os
import sys
import json
import math
import heapq
import torch

# Project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.resnet_posenet import ResNetPoseNet
from training.real_dataset import RealPoseDataset


MODEL_FILE = "results/models/navip_real_chess.pth"
MAP_FILE = "navigation/map/map_nodes.json"
GRAPH_FILE = "navigation/map/navigation_graph.json"

TEST_CSV = "dataset/processed/test.csv"
DATASET_ROOT = "dataset/processed"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ========================================
# Load PoseNet
# ========================================

print("Loading PoseNet...")

model = ResNetPoseNet(pretrained=False)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("PoseNet loaded.")


# ========================================
# Load test image
# ========================================

dataset = RealPoseDataset(
    TEST_CSV,
    DATASET_ROOT
)

image, true_position, true_orientation = dataset[0]

image = image.unsqueeze(0).to(DEVICE)


# ========================================
# Predict pose
# ========================================

with torch.no_grad():

    predicted_position, predicted_orientation = model(
        image
    )

predicted_position = predicted_position[0].cpu()

predicted_xyz = predicted_position.tolist()


# ========================================
# Load map and graph
# ========================================

with open(MAP_FILE, "r") as file:
    map_nodes = json.load(file)

with open(GRAPH_FILE, "r") as file:
    graph = json.load(file)


# ========================================
# Find nearest map node
# ========================================

nearest_node = None
nearest_distance = float("inf")

for node in map_nodes:

    if node["sequence"] != "seq-03":
        continue

    p = node["position"]

    d = math.sqrt(
        (predicted_xyz[0] - p["x"]) ** 2 +
        (predicted_xyz[1] - p["y"]) ** 2 +
        (predicted_xyz[2] - p["z"]) ** 2
    )

    if d < nearest_distance:

        nearest_distance = d
        nearest_node = node


current_node = int(nearest_node["id"])


# ========================================
# Destination
# ========================================

destination_node = 550


# ========================================
# Shortest path
# ========================================

queue = [(0.0, current_node)]

distances = {
    current_node: 0.0
}

previous = {}

while queue:

    current_distance, current = heapq.heappop(queue)

    if current == destination_node:
        break

    for edge in graph["edges"].get(
        str(current),
        []
    ):

        neighbor = int(edge["node"])

        new_distance = (
            current_distance
            + edge["distance"]
        )

        if (
            neighbor not in distances
            or new_distance < distances[neighbor]
        ):

            distances[neighbor] = new_distance

            previous[neighbor] = current

            heapq.heappush(
                queue,
                (new_distance, neighbor)
            )


# ========================================
# Reconstruct route
# ========================================

path = []

current = destination_node

if current == current_node:

    path = [current_node]

elif current not in previous:

    path = []

else:

    while current != current_node:

        path.append(current)

        current = previous[current]

    path.append(current_node)

    path.reverse()


# ========================================
# Output
# ========================================

print()
print("======================================")
print("NaVIP AI Route Navigation")
print("======================================")

print(
    "Predicted Position:",
    [round(v, 4) for v in predicted_xyz]
)

print(
    "Current Map Node:",
    current_node
)

print(
    "Node Matching Distance:",
    f"{nearest_distance:.4f} m"
)

print(
    "Destination Node:",
    destination_node
)

print()

if path:

    print(
        "Route Nodes:",
        len(path)
    )

    print(
        "Route:",
        path
    )

    print(
        "Route Distance:",
        f"{distances[destination_node]:.3f} m"
    )

else:

    print("No route found.")

print("======================================")