import json
import heapq
import math

GRAPH_FILE = "navigation/map/navigation_graph.json"
POI_FILE = "navigation/map/pois.json"


def load_data():

    with open(GRAPH_FILE, "r") as file:
        graph = json.load(file)

    with open(POI_FILE, "r") as file:
        pois = json.load(file)["pois"]

    return graph, pois


def find_poi(pois, poi_id):

    for poi in pois:

        if poi["id"] == poi_id:
            return poi["node_id"]

    raise ValueError(
        f"POI not found: {poi_id}"
    )


def distance(graph, node_a, node_b):

    a = graph["nodes"][str(node_a)]["position"]
    b = graph["nodes"][str(node_b)]["position"]

    return math.sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2 +
        (a["z"] - b["z"]) ** 2
    )


def shortest_path(graph, start, goal):

    queue = [(0.0, start)]

    distances = {
        str(start): 0.0
    }

    previous = {}

    visited = set()

    while queue:

        current_distance, current = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            break

        for edge in graph["edges"][str(current)]:

            neighbor = edge["node"]

            new_distance = (
                current_distance
                + edge["distance"]
            )

            if (
                str(neighbor) not in distances
                or new_distance
                < distances[str(neighbor)]
            ):

                distances[str(neighbor)] = new_distance

                previous[neighbor] = current

                heapq.heappush(
                    queue,
                    (new_distance, neighbor)
                )

    if goal not in previous and start != goal:

        return []

    path = [goal]

    current = goal

    while current != start:

        current = previous[current]

        path.append(current)

    path.reverse()

    return path


if __name__ == "__main__":

    graph, pois = load_data()

    start_node = find_poi(
        pois,
        "poi_01"
    )

    destination_node = find_poi(
        pois,
        "poi_04"
    )

    path = shortest_path(
        graph,
        start_node,
        destination_node
    )

    print("======================================")
    print("NaVIP Route Planner")
    print("======================================")

    print(
        "Start node:",
        start_node
    )

    print(
        "Destination node:",
        destination_node
    )

    if not path:

        print("No route found.")

    else:

        print(
            "Route nodes:",
            len(path)
        )

        print(
            "Path:",
            path
        )

        total_distance = 0.0

        for i in range(len(path) - 1):

            total_distance += distance(
                graph,
                path[i],
                path[i + 1]
            )

        print(
            f"Route distance: "
            f"{total_distance:.3f} m"
        )

    print("======================================")