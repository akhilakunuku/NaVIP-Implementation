"""
map_visualizer.py - Indoor Floor-Plan & Map Visualizer

Displays floor-plan boundaries, rooms, doors, and POIs for Chess or Office datasets.
Outputs clean ASCII representation or saves a Matplotlib visual plot.
"""

import os
import json
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


def load_floorplan(dataset_name="chess"):
    filepath = os.path.join(
        PROJECT_ROOT,
        "navigation",
        "floorplan",
        f"{dataset_name}_floorplan.json"
    )
    if not os.path.isfile(filepath):
        print(f"ERROR: Floor-plan JSON missing: {filepath}")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def render_ascii_map(floorplan_data):
    if not floorplan_data:
        return
    meta = floorplan_data.get("metadata", {})
    print("==================================================")
    print(f"INDOOR FLOOR PLAN — {meta.get('dataset', '').upper()}")
    print("==================================================")
    print(f"Map Type:    {meta.get('map_type')}")
    print(f"Disclaimer:  {meta.get('disclaimer')}")
    print(f"Units:       {meta.get('units')}\n")

    print("ROOMS & SPACES:")
    for room in floorplan_data.get("rooms", []):
        bbox = room.get("bounding_box_2d", {})
        print(f"  • Room [{room['id']}]: {room['name']} ({room['type']})")
        print(f"    Bounds (x: {bbox.get('min_x')} to {bbox.get('max_x')}, y: {bbox.get('min_y')} to {bbox.get('max_y')})")

    print("\nDOORS & PORTALS:")
    for door in floorplan_data.get("doors", []):
        pos = door.get("position", {})
        print(f"  • Door [{door['id']}]: {door['name']} at ({pos.get('x')}, {pos.get('y')}, {pos.get('z')})")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize floor plan.")
    parser.add_argument("--dataset", type=str, default="chess", choices=["chess", "office"])
    args = parser.parse_args()

    fp = load_floorplan(args.dataset)
    render_ascii_map(fp)
