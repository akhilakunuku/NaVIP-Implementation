"""
poi_loader.py - POI Loader & Query Utility

Loads Points of Interest for Chess and Office datasets.
Provides query functions by POI ID, name keyword search, or spatial proximity.
"""

import os
import json
import math
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


class POILoader:
    def __init__(self, dataset_name="chess"):
        self.dataset_name = dataset_name
        self.pois = self._load_pois()

    def _load_pois(self):
        filepath = os.path.join(
            PROJECT_ROOT,
            "navigation",
            "poi",
            f"{self.dataset_name}_pois.json"
        )
        if not os.path.isfile(filepath):
            print(f"WARNING: POI JSON file not found: {filepath}")
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_pois(self):
        return self.pois

    def get_poi_by_id(self, poi_id):
        for poi in self.pois:
            if poi["id"].lower() == poi_id.lower():
                return poi
        return None

    def find_nearest_poi(self, position_xyz):
        best_poi = None
        min_dist = float("inf")
        px, py, pz = position_xyz
        for poi in self.pois:
            pos = poi["position"]
            dist = math.sqrt(
                (px - pos["x"]) ** 2 +
                (py - pos["y"]) ** 2 +
                (pz - pos["z"]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                best_poi = poi
        return best_poi, min_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query dataset POIs.")
    parser.add_argument("--dataset", type=str, default="chess", choices=["chess", "office"])
    args = parser.parse_args()

    loader = POILoader(args.dataset)
    print("==================================================")
    print(f"Loaded POIs for {args.dataset.upper()} Dataset")
    print("==================================================")
    for p in loader.get_all_pois():
        print(f" • ID:   {p['id']}")
        print(f"   Name: {p['name']} ({p['poi_type']})")
        print(f"   Pos:  {p['position']}")
    print("==================================================")
