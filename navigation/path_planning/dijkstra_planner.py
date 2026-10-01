"""
dijkstra_planner.py - Shortest-Path Navigation Planner

Implements Dijkstra's algorithm to calculate shortest paths on the navigation graph.
Input: Current map position (or nearest start node) + Destination POI (or target node).
Output: Ordered path nodes, total path distance, step coordinates.
"""

import os
import sys
import heapq
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "path_planning"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "poi"))

from navigation_graph import NavigationGraph
from poi_loader import POILoader


class DijkstraPlanner:
    def __init__(self, dataset_name="chess"):
        self.dataset_name = dataset_name
        self.graph = NavigationGraph(dataset_name)
        self.poi_loader = POILoader(dataset_name)

    def plan_path_by_nodes(self, start_node_id, target_node_id):
        """
        Dijkstra shortest path algorithm between two node IDs.
        Returns: (path_node_ids, total_distance)
        """
        if start_node_id not in self.graph.nodes or target_node_id not in self.graph.nodes:
            print(f"ERROR: Start node '{start_node_id}' or target node '{target_node_id}' invalid.")
            return [], float("inf")

        distances = {node_id: float("inf") for node_id in self.graph.nodes}
        previous = {node_id: None for node_id in self.graph.nodes}
        distances[start_node_id] = 0.0

        # Priority queue: (current_dist, node_id)
        pq = [(0.0, start_node_id)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node == target_node_id:
                break

            if current_dist > distances[current_node]:
                continue

            for neighbor, weight in self.graph.adjacency.get(current_node, []):
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        # Reconstruct path
        path = []
        curr = target_node_id
        while curr is not None:
            path.append(curr)
            curr = previous[curr]

        path.reverse()
        if path and path[0] == start_node_id:
            return path, distances[target_node_id]
        else:
            return [], float("inf")

    def plan_path_to_poi(self, current_pos_xyz, poi_id):
        """
        Plans path from current 3D position to target POI.
        """
        poi = self.poi_loader.get_poi_by_id(poi_id)
        if not poi:
            print(f"ERROR: Target POI '{poi_id}' not found.")
            return None

        poi_pos = [poi["position"]["x"], poi["position"]["y"], poi["position"]["z"]]
        start_node_id, dist_to_start = self.graph.find_nearest_node(current_pos_xyz)
        target_node_id, dist_to_target = self.graph.find_nearest_node(poi_pos)

        path_nodes, graph_distance = self.plan_path_by_nodes(start_node_id, target_node_id)
        total_distance = dist_to_start + graph_distance + dist_to_target

        path_coords = [self.graph.nodes[nid]["position"] for nid in path_nodes]

        return {
            "dataset": self.dataset_name,
            "target_poi": poi,
            "start_node_id": start_node_id,
            "target_node_id": target_node_id,
            "path_node_ids": path_nodes,
            "path_coordinates": path_coords,
            "total_distance_meters": total_distance
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Dijkstra path planning.")
    parser.add_argument("--dataset", type=str, default="chess", choices=["chess", "office"])
    parser.add_argument("--poi_id", type=str, default="poi_chess_corner_b")
    args = parser.parse_args()

    planner = DijkstraPlanner(args.dataset)
    # Sample starting position near seq-01
    sample_start = [-0.12, -1.12, -0.98]
    result = planner.plan_path_to_poi(sample_start, args.poi_id)

    if result:
        print("==================================================")
        print(f"Dijkstra Route Planning — {args.dataset.upper()}")
        print("==================================================")
        print(f"Destination POI: {result['target_poi']['name']} [{result['target_poi']['id']}]")
        print(f"Start Node:      {result['start_node_id']}")
        print(f"Target Node:     {result['target_node_id']}")
        print(f"Path Length:     {len(result['path_node_ids'])} nodes")
        print(f"Total Distance:  {result['total_distance_meters']:.4f} meters")
        print("==================================================")
