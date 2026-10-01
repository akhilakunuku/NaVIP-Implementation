"""
Stanford 3D Geometry Loader and Inspector.
Inspects PLY and OBJ 3D point cloud/mesh files, computes coordinate bounds
(X min/max, Y min/max, Z min/max), and reports element counts.
"""

import os
import glob
import argparse
import numpy as np


def inspect_ply_geometry(ply_path, sample_limit=50000):
    """
    Parse PLY file header and compute 3D coordinate bounding box.
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    num_vertices = 0
    header_end = False
    vertices = []

    with open(ply_path, 'r', encoding='latin1') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("element vertex"):
                num_vertices = int(line_str.split()[-1])
            if line_str == "end_header":
                header_end = True
                break

        if not header_end:
            raise ValueError(f"Invalid PLY header in {ply_path}")

        # Read sample or all vertices
        count = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    vertices.append([x, y, z])
                    count += 1
                    if sample_limit and count >= sample_limit:
                        break
                except ValueError:
                    continue

    vertices = np.array(vertices, dtype=np.float64)
    if len(vertices) == 0:
        return {
            "file": ply_path,
            "num_vertices": num_vertices,
            "bounds": None
        }

    x_min, y_min, z_min = np.min(vertices, axis=0)
    x_max, y_max, z_max = np.max(vertices, axis=0)

    bounds = {
        "x_min": float(x_min), "x_max": float(x_max),
        "y_min": float(y_min), "y_max": float(y_max),
        "z_min": float(z_min), "z_max": float(z_max)
    }

    return {
        "file": ply_path,
        "total_declared_vertices": num_vertices,
        "sampled_vertices": len(vertices),
        "bounds": bounds
    }


def inspect_area_geometry(dataset_dir="dataset/Stanford2D3D"):
    print("=" * 60)
    print(f"Inspecting Stanford 3D Geometry in: {dataset_dir}")
    print("=" * 60)

    ply_files = glob.glob(os.path.join(dataset_dir, "**", "*.ply"), recursive=True)
    obj_files = glob.glob(os.path.join(dataset_dir, "**", "*.obj"), recursive=True)

    print(f"Found {len(ply_files)} PLY files and {len(obj_files)} OBJ files.")

    all_file_paths = ply_files + obj_files

    for fpath in all_file_paths:
        rel_path = os.path.relpath(fpath, dataset_dir)
        print(f"\nGeometry File: {rel_path}")
        if fpath.endswith('.ply'):
            info = inspect_ply_geometry(fpath)
            print(f"  Declared Vertices: {info['total_declared_vertices']}")
            if info['bounds']:
                b = info['bounds']
                print(f"  X Bounds: [{b['x_min']:.2f}, {b['x_max']:.2f}] meters")
                print(f"  Y Bounds: [{b['y_min']:.2f}, {b['y_max']:.2f}] meters")
                print(f"  Z Bounds: [{b['z_min']:.2f}, {b['z_max']:.2f}] meters")
        else:
            print("  OBJ mesh file discovered.")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Stanford 3D Geometry")
    parser.add_argument("--dataset-dir", type=str, default="dataset/Stanford2D3D")
    args = parser.parse_args()

    inspect_area_geometry(args.dataset_dir)
