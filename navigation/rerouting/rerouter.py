"""
rerouter.py - Dynamic Path Rerouting Manager

Triggers Dijkstra shortest path recalculation from current off-route location to target POI.
Generates updated voice instructions for the new route.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "path_planning"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "voice"))

from dijkstra_planner import DijkstraPlanner
from instruction_generator import InstructionGenerator
from voice_output import VoiceOutputManager


def execute_reroute(dataset_name, current_position, target_poi_id, planner=None):
    if planner is None:
        planner = DijkstraPlanner(dataset_name)

    print(f"\n[REROUTER]: Recalculating path from current off-route location {current_position} to '{target_poi_id}'...")

    new_route = planner.plan_path_to_poi(current_position, target_poi_id)
    if not new_route or not new_route["path_coordinates"]:
        print("ERROR: Dynamic rerouting failed to find a valid graph path.")
        return None

    instructor = InstructionGenerator()
    voice = VoiceOutputManager(enable_tts=False)

    target_name = new_route["target_poi"]["name"]
    first_waypoint = new_route["path_coordinates"][0]

    voice.speak("Route deviation detected. Recalculating new path...")
    inst = instructor.generate_turn_instruction(current_position, first_waypoint)
    voice.speak(f"New path ready to {target_name}. {inst}")

    return new_route


if __name__ == "__main__":
    off_route_pos = [0.1, 0.85, 0.0]
    new_route = execute_reroute("chess", off_route_pos, "poi_chess_corner_b")
    if new_route:
        print("==================================================")
        print("Dynamic Rerouting Test Successful")
        print(f"New Path Length: {len(new_route['path_node_ids'])} nodes")
        print("==================================================")
