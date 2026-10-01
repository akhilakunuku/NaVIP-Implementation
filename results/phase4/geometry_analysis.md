# Area 3 3D Environment Analysis

## Environmental Structure & Derived Elements
1. **Floors**: Floor planes are horizontally aligned along $Z \approx 0.0m$ to $Z \approx 0.2m$. Camera height is consistently positioned at camera elevation $Z \approx 1.38m$.
2. **Walls & Boundaries**: Structural walls enclose each room. Rooms span spatial extents: $X \in [2.0, 26.5]m$, $Y \in [-12.5, 4.0]m$.
3. **Rooms & Corridors**: Area 3 consists of 21 distinct rooms and interconnecting hallways (`hallway_1_3`, `hallway_2_3`, `hallway_3_3`, `hallway_4_3`, `hallway_6_3`, `office_1` through `10`, `lounge_1` and `2`, `conferenceRoom_1`, `WC_1` and `2`, `storage_1`).
4. **Doors & Openings**: Open passages connect offices and lounges to central hallways.
5. **Walkable Surfaces**: The 2D projection of camera trajectory points $(X, Y)$ along with bounding room polygons defines the walkable 2D floorplan space.
