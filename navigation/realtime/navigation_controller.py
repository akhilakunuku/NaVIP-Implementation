"""
navigation_controller.py - Continuous Re-localization & Navigation Controller

Active navigation state machine accepting frame-by-frame 6-DoF pose predictions:
- Tracks progress along planned route waypoints.
- Determines step progression ("CONTINUE", "NEXT_WAYPOINT", "ARRIVED", "DEVIATED").
- Triggers dynamic rerouting if pose strays off route.
"""

import os
import sys
import math

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "path_planning"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "voice"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "rerouting"))

from dijkstra_planner import DijkstraPlanner
from instruction_generator import InstructionGenerator
from voice_output import VoiceOutputManager


class RealtimeNavigationController:
    def __init__(self, dataset_name="chess", target_poi_id=None, deviation_threshold=0.4, waypoint_reach_dist=0.3):
        self.dataset_name = dataset_name
        self.target_poi_id = target_poi_id
        self.deviation_threshold = deviation_threshold
        self.waypoint_reach_dist = waypoint_reach_dist

        self.planner = DijkstraPlanner(dataset_name)
        self.instructor = InstructionGenerator()
        self.voice = VoiceOutputManager(enable_tts=False)

        self.current_route = None
        self.current_waypoint_idx = 0
        self.is_active = False

    def start_navigation(self, initial_position, target_poi_id):
        """Initializes route planning from start position to POI."""
        self.target_poi_id = target_poi_id
        self.current_route = self.planner.plan_path_to_poi(initial_position, target_poi_id)

        if not self.current_route or not self.current_route["path_coordinates"]:
            print("ERROR: Route planning failed.")
            self.is_active = False
            return False

        self.current_waypoint_idx = 0
        self.is_active = True

        target_name = self.current_route["target_poi"]["name"]
        self.voice.speak(f"Starting navigation to {target_name}. Route planned ({self.current_route['total_distance_meters']:.1f} meters).")

        first_waypoint = self.current_route["path_coordinates"][0]
        inst = self.instructor.generate_turn_instruction(initial_position, first_waypoint)
        self.voice.speak(inst)
        return True

    def process_pose_update(self, predicted_position, predicted_orientation=None):
        """
        Accepts a continuous 6-DoF predicted pose from PoseNet inference.
        Evaluates step state: CONTINUE, NEXT_WAYPOINT, ARRIVED, or DEVIATED.
        """
        if not self.is_active or not self.current_route:
            return {"status": "INACTIVE", "message": "Navigation not active."}

        coords = self.current_route["path_coordinates"]
        target_poi_pos = [
            self.current_route["target_poi"]["position"]["x"],
            self.current_route["target_poi"]["position"]["y"],
            self.current_route["target_poi"]["position"]["z"]
        ]

        # 1. Check distance to final POI target
        dist_to_final = math.sqrt(
            (predicted_position[0] - target_poi_pos[0])**2 +
            (predicted_position[1] - target_poi_pos[1])**2 +
            (predicted_position[2] - target_poi_pos[2])**2
        )

        if dist_to_final <= self.waypoint_reach_dist:
            self.is_active = False
            target_name = self.current_route["target_poi"]["name"]
            msg = f"Destination reached. You have arrived at {target_name}."
            self.voice.speak(msg)
            return {"status": "ARRIVED", "message": msg}

        # 2. Check distance to current waypoint
        current_waypoint = coords[self.current_waypoint_idx]
        dist_to_waypoint = math.sqrt(
            (predicted_position[0] - current_waypoint[0])**2 +
            (predicted_position[1] - current_waypoint[1])**2 +
            (predicted_position[2] - current_waypoint[2])**2
        )

        if dist_to_waypoint <= self.waypoint_reach_dist and self.current_waypoint_idx < len(coords) - 1:
            self.current_waypoint_idx += 1
            next_waypoint = coords[self.current_waypoint_idx]
            turn_inst = self.instructor.generate_turn_instruction(predicted_position, next_waypoint, prev_pos=current_waypoint)
            self.voice.speak(turn_inst)
            return {"status": "NEXT_WAYPOINT", "waypoint_idx": self.current_waypoint_idx, "message": turn_inst}

        # 3. Check deviation from path line segment
        from deviation_detector import compute_perpendicular_distance
        if self.current_waypoint_idx < len(coords) - 1:
            seg_start = coords[self.current_waypoint_idx]
            seg_end = coords[self.current_waypoint_idx + 1]
            dev_dist = compute_perpendicular_distance(predicted_position, seg_start, seg_end)
        else:
            dev_dist = dist_to_waypoint

        if dev_dist > self.deviation_threshold:
            print(f"\n[DEVIATION DETECTED]: Distance to route line is {dev_dist:.2f}m (Threshold: {self.deviation_threshold:.2f}m).")
            from rerouter import execute_reroute
            new_route = execute_reroute(self.dataset_name, predicted_position, self.target_poi_id, self.planner)
            if new_route:
                self.current_route = new_route
                self.current_waypoint_idx = 0
                return {"status": "REROUTED", "message": "Route recalculated successfully."}
            else:
                return {"status": "REROUTE_FAILED", "message": "Failed to recalculate route."}

        # 4. Standard path continuation
        poi_announcement = self.instructor.generate_poi_announcement(
            predicted_position,
            self.current_route["target_poi"]["name"],
            target_poi_pos
        )
        return {
            "status": "CONTINUE",
            "current_waypoint_idx": self.current_waypoint_idx,
            "distance_to_target": dist_to_final,
            "poi_announcement": poi_announcement
        }


if __name__ == "__main__":
    controller = RealtimeNavigationController(dataset_name="chess")
    init_pos = [-0.12, -1.12, -0.98]
    controller.start_navigation(init_pos, "poi_chess_corner_b")

    # Simulate 3 continuous pose updates
    p1 = [-0.12, -1.10, -0.97]
    state1 = controller.process_pose_update(p1)

    print("==================================================")
    print("Realtime Navigation Controller Simulation")
    print("==================================================")
    print(f"Step 1 State: {state1['status']}")
    print(f"Announcement: {state1.get('poi_announcement')}")
    print("==================================================")
