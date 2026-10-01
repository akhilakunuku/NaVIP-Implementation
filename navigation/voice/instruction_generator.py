"""
instruction_generator.py - Voice Instruction Generator

Converts planned paths into natural spoken turn-by-turn guidance and NaVIP relative clock-position announcements.

Examples:
- "Move forward."
- "Turn left."
- "Turn right."
- "Continue for 3.5 meters."
- "You are approaching Queen Side Board Corner."
- "Destination reached."
- "Queen Side Board Corner is at your 2 o'clock position, 1.5 meters away."
"""

import math
import numpy as np


class InstructionGenerator:
    def __init__(self):
        pass

    def calculate_heading_angle(self, p1, p2):
        """Calculates 2D yaw heading angle in degrees from p1 to p2 on XY plane."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        heading = math.degrees(math.atan2(dy, dx))
        return (heading + 360) % 360

    def generate_turn_instruction(self, current_pos, next_pos, prev_pos=None):
        """Generates turn instruction based on 3-point direction change."""
        dist = math.sqrt(
            (next_pos[0] - current_pos[0])**2 +
            (next_pos[1] - current_pos[1])**2 +
            (next_pos[2] - current_pos[2])**2
        )

        if prev_pos is None:
            return f"Move forward for {dist:.1f} meters."

        heading_in = self.calculate_heading_angle(prev_pos, current_pos)
        heading_out = self.calculate_heading_angle(current_pos, next_pos)

        turn_angle = (heading_out - heading_in + 360) % 360

        if turn_angle < 30 or turn_angle > 330:
            return f"Continue straight for {dist:.1f} meters."
        elif 30 <= turn_angle < 150:
            return f"Turn left and proceed for {dist:.1f} meters."
        elif 150 <= turn_angle <= 210:
            return f"Make a U-turn and proceed for {dist:.1f} meters."
        else:
            return f"Turn right and proceed for {dist:.1f} meters."

    def generate_poi_announcement(self, current_pos, poi_name, poi_pos, user_heading_deg=0.0):
        """
        Generates NaVIP relative clock-position announcement:
        "[POI name] is at your [clock position] o'clock, [distance] meters away."
        """
        dx = poi_pos[0] - current_pos[0]
        dy = poi_pos[1] - current_pos[1]
        dist = math.sqrt(dx**2 + dy**2 + (poi_pos[2] - current_pos[2])**2)

        target_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        relative_angle = (target_angle - user_heading_deg + 360) % 360

        # Convert relative angle (0-360) to clock hour (1-12)
        clock_hour = int(round(relative_angle / 30.0)) % 12
        if clock_hour == 0:
            clock_hour = 12

        return f"[{poi_name}] is at your {clock_hour} o'clock position, {dist:.1f} meters away."


if __name__ == "__main__":
    generator = InstructionGenerator()
    p1 = [0.0, 0.0, 0.0]
    p2 = [1.0, 0.0, 0.0]
    p3 = [1.0, 1.0, 0.0]

    turn_msg = generator.generate_turn_instruction(p2, p3, prev_pos=p1)
    poi_msg = generator.generate_poi_announcement(p1, "Queen Side Corner", [1.0, 1.0, 0.0], user_heading_deg=0.0)

    print("==================================================")
    print("Voice Instruction Generator Output Test")
    print("==================================================")
    print(f"Turn Instruction: {turn_msg}")
    print(f"POI Announcement: {poi_msg}")
    print("==================================================")
