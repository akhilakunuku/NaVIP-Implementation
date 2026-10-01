# NaVIP Point of Interest (POI) Schema Specification

This document defines the schema for target destination POIs used for path planning and audio announcements in NaVIP.

## Fields
- **`id`**: String unique identifier (e.g. `"poi_queen_side"`, `"poi_conference_table"`).
- **`name`**: Human-readable spoken destination name.
- **`dataset`**: Dataset association (`"chess"` or `"office"`).
- **`poi_type`**: Type tag (`"DEMO / APPROXIMATE POI"`).
- **`floor_level`**: Integer floor number.
- **`position`**: 3D spatial coordinate `{"x": float, "y": float, "z": float}`.
- **`entrance_position`**: 3D coordinate for door/waypoint access.
- **`priority`**: Integer priority ranking for search sorting.
- **`description`**: Detailed context for voice guidance announcements.
