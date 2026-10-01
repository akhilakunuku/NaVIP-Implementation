import json
import math
from voice_guidance import VoiceGuidance

GRAPH_FILE = "navigation/map/navigation_graph.json"


def load_graph():

    with open(GRAPH_FILE, "r") as file:
        return json.load(file)


def get_position(graph, node_id):

    return graph["nodes"][str(node_id)]["position"]


def direction_vector(a, b):

    return (
        b["x"] - a["x"],
        b["y"] - a["y"],
        b["z"] - a["z"]
    )


def horizontal_angle(v):

    # Use X-Z plane for indoor movement direction
    return math.degrees(
        math.atan2(v[2], v[0])
    )


def normalize_angle(angle):

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def classify_turn(angle):

    if abs(angle) < 20:
        return "Continue straight"

    if 20 <= angle < 60:
        return "Slight right"

    if 60 <= angle < 120:
        return "Turn right"

    if angle >= 120:
        return "Sharp right"

    if -60 < angle <= -20:
        return "Slight left"

    if -120 < angle <= -60:
        return "Turn left"

    return "Sharp left"


def generate_instructions(graph, route):

    instructions = []

    if len(route) < 2:
        return instructions

    # First movement
    previous_vector = direction_vector(
        get_position(graph, route[0]),
        get_position(graph, route[1])
    )

    instructions.append(
        f"Move from node {route[0]} to node {route[1]}"
    )

    for i in range(1, len(route) - 1):

        current = route[i]
        next_node = route[i + 1]

        current_position = get_position(
            graph,
            current
        )

        next_position = get_position(
            graph,
            next_node
        )

        current_vector = direction_vector(
            current_position,
            next_position
        )

        previous_angle = horizontal_angle(
            previous_vector
        )

        current_angle = horizontal_angle(
            current_vector
        )

        turn_angle = normalize_angle(
            current_angle - previous_angle
        )

        instruction = classify_turn(
            turn_angle
        )

        instructions.append(
            f"{instruction} at node {current} "
            f"then continue to node {next_node}"
        )

        previous_vector = current_vector

    instructions.append(
        f"Destination reached at node {route[-1]}"
    )

    return instructions


if __name__ == "__main__":

    graph = load_graph()
    voice = VoiceGuidance()

    route = [
        436, 439, 441, 443, 446, 449,
        453, 456, 459, 462, 463, 465,
        467, 469, 473, 476, 477, 480,
        482, 483, 486, 488, 490, 494,
        498, 500, 503, 507, 508, 513,
        516, 517, 520, 524, 531, 532,
        536, 540, 546, 550
    ]

    instructions = generate_instructions(
        graph,
        route
    )

    print()
    print("======================================")
    print("NaVIP Turn-by-Turn Navigation")
    print("======================================")

    for i, instruction in enumerate(
        instructions,
        start=1
    ):

        print(
            f"{i}. {instruction}"
        )
        voice.speak(instruction)

    print("======================================")