import json
import math

MAP_FILE = "navigation/map/map_nodes.json"
GRAPH_FILE = "navigation/map/navigation_graph.json"

# Maximum distance between two nodes to create an edge
MAX_DISTANCE = 0.08

with open(MAP_FILE, "r") as file:
    nodes = json.load(file)

graph = {
    "nodes": {},
    "edges": {}
}

# Store nodes
for node in nodes:
    node_id = str(node["id"])

    graph["nodes"][node_id] = {
        "sequence": node["sequence"],
        "position": node["position"],
        "image": node["image"]
    }

    graph["edges"][node_id] = []


# Connect nearby nodes belonging to the same sequence
for i in range(len(nodes)):

    node_a = nodes[i]

    for j in range(i + 1, len(nodes)):

        node_b = nodes[j]

        # Do not connect different sequences
        if node_a["sequence"] != node_b["sequence"]:
            continue

        ax = node_a["position"]["x"]
        ay = node_a["position"]["y"]
        az = node_a["position"]["z"]

        bx = node_b["position"]["x"]
        by = node_b["position"]["y"]
        bz = node_b["position"]["z"]

        distance = math.sqrt(
            (ax - bx) ** 2 +
            (ay - by) ** 2 +
            (az - bz) ** 2
        )

        if distance <= MAX_DISTANCE:

            a = str(node_a["id"])
            b = str(node_b["id"])

            graph["edges"][a].append({
                "node": int(b),
                "distance": distance
            })

            graph["edges"][b].append({
                "node": int(a),
                "distance": distance
            })


with open(GRAPH_FILE, "w") as file:
    json.dump(graph, file, indent=2)


edge_count = sum(
    len(edges)
    for edges in graph["edges"].values()
) // 2

print("======================================")
print("NaVIP Navigation Graph")
print("======================================")
print("Nodes:", len(graph["nodes"]))
print("Edges:", edge_count)
print("Output:", GRAPH_FILE)
print("======================================")