# Stanford 2D-3D-S Area 3 Pose Coordinate Report

## 1. Camera & Coordinate System Specifications
- **Camera Location (`camera_location`)**: 3D position vector $[x, y, z]$ in global meters frame of Area 3.
- **Camera Extrinsic Matrix (`camera_rt_matrix`)**: $3 \times 4$ transformation matrix $[R_{cw} \vert t_{cw}]$ converting 3D world coordinates to camera coordinates.
- **Rotation Matrix Transformation**: World-to-camera rotation is $R_{cw}$. The camera-to-world rotation is $R_{wc} = R_{cw}^T$.
- **Quaternion Representation**: Hamilton unit quaternion $(qw, qx, qy, qz)$ normalized to $\|q\|_2 = 1.0$.
- **Position Units**: Meters ($m$).
- **Room Identification**: Each pose JSON explicitly links `room` (e.g. `office_3_3`, `lounge_2_3`, `hallway_1_3`).

## 2. Confirmation of Transpose Formula
$$R_{wc} = R_{cw}^T$$

Confirmed: The camera-to-world rotation matrix is the transpose of the world-to-camera rotation matrix provided in Stanford pose JSON files. Ground-truth pose coordinates directly align with the 3D mesh geometry frame of Area 3.
