# Stanford 2D-3D-S Area 3 3D Data Inventory

Inventory of 3D geometry, mesh, point cloud, and camera mapping files present in `dataset/Stanford2D3D/area_3/3d/`:

| Filename | Type | Purpose | Mesh / Point Cloud / Semantic | Coordinate System | Units | Walkable Space Derivable? |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `camera_to_room.json` | JSON Mapping File | Maps camera UUID string to room identifier (e.g. office_3_3, lounge_1_3) | Metadata / Spatial Index | N/A | N/A | YES |
| `pointcloud.mat` | MATLAB Binary MAT Data File | High-density 3D point cloud of Area 3 containing 3D vertex coordinates, RGB colors, semantic element IDs, and camera positions | Dense 3D Point Cloud with Semantic Labels | Stanford global 3D reference frame (X, Y, Z) | Meters | YES |
| `rgb.obj` | Wavefront 3D OBJ Geometry File | 3D surface mesh containing vertices, texture UV coordinates, and face definitions for visual rendering | 3D Surface Mesh (Textured RGB) | Stanford global 3D reference frame (X, Y, Z) | Meters | YES |
| `rgb.mtl` | Material Library File | Defines material properties and texture mapping references for rgb.obj | Material Metadata | N/A | N/A | NO |
| `semantic.obj` | Wavefront 3D OBJ Geometry File | 3D surface mesh colored by semantic class (floor, wall, door, window, chair, table, ceiling) | 3D Surface Mesh (Semantic Categorization) | Stanford global 3D reference frame (X, Y, Z) | Meters | YES |
| `semantic.mtl` | Material Library File | Defines RGB color codes for semantic categories (e.g., floor, wall, door, window) | Semantic Color Metadata | N/A | N/A | NO |
