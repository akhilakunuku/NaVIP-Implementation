"""
navigation_graph.py - Independent Navigation Graph Representation

Constructs a graph data structure (nodes and weighted walkable edges) for path planning.
Loads map node positions from existing map node JSON files without modifying them.
"""

import os
import json
import math
import heapq
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


class NavigationGraph:
    def __init__(self, dataset_name="chess", max_edge_dist=0.6, max_neighbors=4):
        self.dataset_name = dataset_name
        self.max_edge_dist = max_edge_dist
        self.max_neighbors = max_neighbors
        self.nodes = {}       # node_id -> {"position": [x,y,z], ...}
        self.adjacency = {}   # node_id -> [(neighbor_id, distance), ...]
        self._build_graph()

    def _get_map_file(self):
        if self.dataset_name == "chess":
            return os.path.join(PROJECT_ROOT, "navigation", "map", "map_nodes.json")
        else:
            return os.path.join(PROJECT_ROOT, "navigation", "map", "office_map_nodes.json")

    def _build_graph(self):
        map_file = self._get_map_file()
        if not os.path.isfile(map_file):
            print(f"WARNING: Map file not found: {map_file}")
            return

        with open(map_file, "r", encoding="utf-8") as f:
            raw_nodes = json.load(f)

        # 1. Parse nodes
        for idx, node in enumerate(raw_nodes):
            node_id = str(node.get("id", idx))
            pos = node["position"]
            self.nodes[node_id] = {
                "id": node_id,
                "position": [float(pos["x"]), float(pos["y"]), float(pos["z"])],
                "sequence": node.get("sequence", ""),
                "image": node.get("image", "")
            }
            self.adjacency[node_id] = []

        # 2. Build spatial edges based on Euclidean proximity
        node_keys = list(self.nodes.keys())
        coords = [self.nodes[k]["position"] for k in node_keys]

        # Connect consecutive nodes within sequences
        for i in range(len(node_keys) - 1):
            n1, n2 = node_keys[i], node_keys[i+1]
            p1, p2 = coords[i], coords[i+1]
            dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
            if dist <= self.max_edge_dist * 3.0:
                self.adjacency[n1].append((n2, dist))
                self.adjacency[n2].append((n1, dist))

        # Add cross-sequence spatial bridge edges (sampled every 20 nodes for speed)
        sampled_keys = node_keys[::20]
        for i in range(len(sampled_keys)):
            n1 = sampled_keys[i]
            p1 = self.nodes[n1]["position"]
            for j in range(i + 1, len(sampled_keys)):
                n2 = sampled_keys[j]
                if self.nodes[n1]["sequence"] != self.nodes[n2]["sequence"]:
                    p2 = self.nodes[n2]["position"]
                    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                    if dist <= self.max_edge_dist * 1.5:
                        self.adjacency[n1].append((n2, dist))
                        self.adjacency[n2].append((n1, dist))

    def find_nearest_node(self, pos_xyz):
        """Finds the closest node in graph to a given 3D position."""
        best_id = None
        min_dist = float("inf")
        px, py, pz = pos_xyz
        for node_id, data in self.nodes.items():
            nx, ny, nz = data["position"]
            dist = math.sqrt((px - nx)**2 + (py - ny)**2 + (pz - nz)**2)
            if dist < min_dist:
                min_dist = dist
                best_id = node_id
        return best_id, min_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and inspect navigation graph.")
    parser.add_argument("--dataset", type=str, default="chess", choices=["chess", "office"])
    args = parser.parse_args()

    graph = NavigationGraph(args.dataset)
    print("==================================================")
    print(f"Navigation Graph Built — {args.dataset.upper()}")
    print("==================================================")
    print(f"Total Nodes:     {len(graph.nodes)}")
    print(f"Adjacency Nodes: {len(graph.adjacency)}")
    print("==================================================")
