"""
run_navip_demo.py - Complete End-to-End NaVIP Pipeline Demonstration

Connects the entire NaVIP base-paper pipeline:
1. Input RGB Image (Chess or Office dataset sample)
2. Existing PoseNet model loading & 6-DoF pose prediction (REUSING models.resnet_posenet)
3. Indoor Floor-Plan coordinate alignment
4. Destination POI selection
5. Dijkstra shortest path planning
6. Continuous re-localization & state controller simulation
7. Synthetic deviation insertion & dynamic rerouting trigger
8. Spoken turn-by-turn and clock-position audio guidance

NOTICE:
Static image execution is explicitly demarcated as a single-frame pipeline demonstration.
The controller interface accepts continuous real-time video frames via `process_frame()`.
Original models, checkpoints, CSVs, and scripts remain strictly untouched.
"""

import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------
# Import Existing Unchanged Model Architecture
# ----------------------------------------------------
from models.resnet_posenet import ResNetPoseNet

# ----------------------------------------------------
# Import New NaVIP Modules
# ----------------------------------------------------
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "floorplan"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "poi"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "path_planning"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "voice"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "realtime"))
sys.path.append(os.path.join(PROJECT_ROOT, "navigation", "rerouting"))

from coordinate_alignment import CoordinateAlignmentManager
from destination_selector import select_destination
from dijkstra_planner import DijkstraPlanner
from instruction_generator import InstructionGenerator
from voice_output import VoiceOutputManager
from navigation_controller import RealtimeNavigationController
from deviation_detector import is_deviated
from rerouter import execute_reroute


def load_posenet_model(dataset_name="stanford", device="cpu"):
    """Loads existing trained PoseNet checkpoint without modifying any files."""
    model_path = os.path.join(PROJECT_ROOT, "results", "models", "navip_stanford_best.pth")
    if not os.path.isfile(model_path):
        model_path = os.path.join(PROJECT_ROOT, "results", "models", "navip_stanford_final.pth")

    if not os.path.isfile(model_path):
        print(f"ERROR: Model checkpoint missing: {model_path}")
        return None, None

    model = ResNetPoseNet(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_path


def lookup_ground_truth_pose(dataset_name, image_path):
    """Searches dataset CSV files for the exact image ground-truth 6-DoF pose."""
    import math
    import pandas as pd
    target_filename = os.path.basename(image_path).lower()
    
    csv_candidates = []
    if dataset_name == "chess":
        csv_candidates = [
            os.path.join(PROJECT_ROOT, "dataset", "processed", "train.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "labels", "test.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "labels", "validation.csv")
        ]
    else:
        csv_candidates = [
            os.path.join(PROJECT_ROOT, "dataset", "processed", "office", "train_final.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "processed", "office", "test.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "processed", "office", "validation.csv")
        ]

    for csv_file in csv_candidates:
        if os.path.isfile(csv_file):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    img_str = str(row["image"]).replace("/", os.sep)
                    if os.path.basename(img_str).lower() == target_filename:
                        pos_gt = [float(row["x"]), float(row["y"]), float(row["z"])]
                        ori_gt = [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])]
                        return pos_gt, ori_gt, os.path.basename(csv_file)
            except Exception:
                continue
    return None, None, None


def run_pipeline_demo(dataset_name="chess", image_path=None, destination_poi_id=None, test_deviation=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==================================================================")
    print("           NaVIP BASE PAPER END-TO-END DEMONSTRATION              ")
    print("==================================================================")
    print(f"Dataset Mode: {dataset_name.upper()}")
    print(f"Device:       {device}")

    # 1. Resolve Input Sample Image
    if image_path is None:
        if dataset_name == "chess":
            image_path = os.path.join(PROJECT_ROOT, "dataset", "processed", "test", "seq-03", "seq-03", "frame-000000.color.png")
        else:
            image_path = os.path.join(PROJECT_ROOT, "dataset", "office", "office", "seq-01", "frame-000000.color.png")

    image_path = os.path.abspath(os.path.normpath(image_path))
    print(f"Input Image:  {image_path}\n")

    if not os.path.isfile(image_path):
        print(f"ERROR: Sample image does not exist: {image_path}")
        return

    # 2. Existing PoseNet Inference
    print("[1/6] Running Existing ResNet-18 PoseNet Model Inference...")
    model, loaded_pth = load_posenet_model(dataset_name, device)
    if model is None:
        print("Pipeline aborted due to missing model checkpoint.")
        return

    print(f"Loaded trained checkpoint: {loaded_pth}")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    raw_img = Image.open(image_path).convert("RGB")
    tensor_img = preprocess(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_pos, pred_ori = model(tensor_img)

    pred_pos_xyz = pred_pos[0].cpu().tolist()
    pred_ori_q = pred_ori[0].cpu().tolist()

    # Ground-Truth Pose Lookup
    pos_gt, ori_gt, csv_source = lookup_ground_truth_pose(dataset_name, image_path)

    print(f"\n--- Predicted Pose vs 7-Scenes Ground Truth ---")
    print(f"  • Predicted Position (x,y,z):  {[round(v, 4) for v in pred_pos_xyz]}")
    print(f"  • Predicted Orientation (q):   {[round(v, 4) for v in pred_ori_q]}")
    if pos_gt:
        print(f"  • Ground-Truth Position (x,y,z):{[round(v, 4) for v in pos_gt]} (Source: {csv_source})")
        print(f"  • Ground-Truth Orientation (q): {[round(v, 4) for v in ori_gt]}")
        import math
        pos_err = math.sqrt(sum((p - g)**2 for p, g in zip(pred_pos_xyz, pos_gt)))
        print(f"  • Position Prediction Error:   {pos_err:.4f} meters")
    else:
        print("  • Ground-Truth Pose:            Not found in dataset CSVs.")
    print()

    # 3. Floor-Plan Coordinate Alignment
    print("[2/6] Floor-Plan Coordinate Frame Alignment...")
    aligner = CoordinateAlignmentManager(scale=1.0)
    floorplan_pos = aligner.world_to_floorplan(pred_pos_xyz)
    print(f"  • Floorplan Transformed Pos: {[round(v, 4) for v in floorplan_pos]}\n")

    # 4. Target POI Selection
    print("[3/6] Target Destination Selection...")
    target_poi = select_destination(dataset_name, destination_poi_id)
    if not target_poi:
        print("ERROR: POI selection failed.")
        return

    print(f"  • Target POI: {target_poi['name']} [{target_poi['id']}]")
    print(f"  • POI Type:   {target_poi['poi_type']}\n")

    # 5. Dijkstra Shortest-Path Planning
    print("[4/6] Running Dijkstra Path Planning...")
    planner = DijkstraPlanner(dataset_name)
    route = planner.plan_path_to_poi(pred_pos_xyz, target_poi["id"])

    if not route or not route["path_coordinates"]:
        print("ERROR: Dijkstra route planning failed.")
        return

    print(f"  • Start Node ID: {route['start_node_id']}")
    print(f"  • Target Node ID:{route['target_node_id']}")
    print(f"  • Route Nodes:   {len(route['path_node_ids'])} nodes")
    print(f"  • Total Distance:{route['total_distance_meters']:.3f} meters\n")

    # 6. Spoken Voice Guidance Generation
    print("[5/6] Spoken Voice Guidance Announcement...")
    voice = VoiceOutputManager(enable_tts=True)
    instructor = InstructionGenerator()

    voice.speak(f"NaVIP system initialized. Target destination: {target_poi['name']}.")
    first_wp = route["path_coordinates"][0]
    first_inst = instructor.generate_turn_instruction(pred_pos_xyz, first_wp)
    voice.speak(first_inst)

    poi_announcement = instructor.generate_poi_announcement(
        pred_pos_xyz, target_poi["name"],
        [target_poi["position"]["x"], target_poi["position"]["y"], target_poi["position"]["z"]]
    )
    voice.speak(poi_announcement)

    # 7. Continuous Re-localization & Synthetic Deviation Rerouting Test
    print("\n[6/6] Continuous Re-localization & Deviation Rerouting Test...")
    controller = RealtimeNavigationController(dataset_name, target_poi_id=target_poi["id"])
    controller.start_navigation(pred_pos_xyz, target_poi["id"])

    if test_deviation:
        print("\n--- Simulating Off-Route Motion (Deviation Test) ---")
        # Shift position off route by 0.9m
        off_route_xyz = [pred_pos_xyz[0] + 0.8, pred_pos_xyz[1] + 0.5, pred_pos_xyz[2]]
        step_result = controller.process_pose_update(off_route_xyz)
        print(f"Step Result: {step_result['status']}")

    print("\n==================================================================")
    print("SUCCESS: NaVIP End-to-End Pipeline Demonstration Completed.")
    print("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete NaVIP end-to-end demonstration.")
    parser.add_argument("--dataset", type=str, default="chess", choices=["chess", "office"])
    parser.add_argument("--image", type=str, default=None, help="Input RGB image path")
    parser.add_argument("--destination", type=str, default=None, help="Target POI ID")
    parser.add_argument("--no_deviation_test", action="store_true", help="Disable synthetic deviation test")

    args = parser.parse_args()
    run_pipeline_demo(
        dataset_name=args.dataset,
        image_path=args.image,
        destination_poi_id=args.destination,
        test_deviation=not args.no_deviation_test
    )
