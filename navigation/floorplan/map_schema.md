# NaVIP Indoor Floor-Plan Schema Specification

This document defines the schema for indoor floor-plan representations used by the NaVIP navigation pipeline.

## Metadata Header
- `dataset`: String (`"chess"` or `"office"`)
- `map_type`: String (`"DEMO / APPROXIMATE MAP"`)
- `version`: String (`"1.0"`)
- `units`: String (`"meters"`)

## Structural Components
1. **`rooms`**: Array of room objects (`id`, `name`, `type`, `polygon_2d`, `floor_level`).
2. **`corridors`**: Array of corridor objects connecting rooms.
3. **`doors`**: Array of door/entrance portals connecting rooms and walkable paths.
4. **`walkable_areas`**: Polygon regions specifying accessible floor space.
5. **`nodes`**: Topological map nodes (`id`, `position`: `{x, y, z}`, `room_id`, `node_type`).
6. **`edges`**: Accessible graph connections between nodes (`source`, `target`, `distance`, `walkable`).
7. **`pois`**: Points of interest located on the floor plan (`id`, `name`, `position`, `room_id`).
