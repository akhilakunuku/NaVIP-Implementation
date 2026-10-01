"""
destination_selector.py - Interactive & Automated Destination Selector

Allows users or upper-level controllers to select a destination POI for path planning.
"""

import argparse
from poi_loader import POILoader


def select_destination(dataset_name="chess", poi_id=None):
    loader = POILoader(dataset_name)
    pois = loader.get_all_pois()

    if not pois:
        print("ERROR: No POIs available.")
        return None

    if poi_id:
        selected = loader.get_poi_by_id(poi_id)
        if selected:
            return selected
        print(f"WARNING: POI '{poi_id}' not found. Defaulting to first available POI.")

    # Default selection
    return pois[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select destination POI.")
    parser.add_argument("--dataset", type=str, default="chess", choices=["chess", "office"])
    parser.add_argument("--poi_id", type=str, default=None, help="Target POI ID")
    args = parser.parse_args()

    target = select_destination(dataset_name=args.dataset, poi_id=args.poi_id)
    if target:
        print("==================================================")
        print(f"Selected Target Destination: {target['name']} [{target['id']}]")
        print(f"Position: {target['position']}")
        print(f"Type:     {target['poi_type']}")
        print("==================================================")
