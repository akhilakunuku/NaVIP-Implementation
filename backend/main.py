import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# -----------------------------------------------------------------------------
# Path & Environment Setup
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.resnet_posenet import ResNetPoseNet

# -----------------------------------------------------------------------------
# FastAPI App & CORS Setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="NaVIP Backend API",
    description="Backend services for NaVIP Pose Estimation & Navigation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global Paths & Model Instance
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = ROOT_DIR / "results" / "models" / "navip_stanford_best.pth"
MAP_POINTS_PATH = ROOT_DIR / "results" / "phase4" / "map" / "area3_map_points.csv"
MAP_ROOMS_PATH = ROOT_DIR / "results" / "phase4" / "map" / "area3_rooms.json"
EVAL_SUMMARY_PATH = ROOT_DIR / "results" / "evaluation" / "stanford_test_summary.json"
EVAL_RESULTS_PATH = ROOT_DIR / "results" / "evaluation" / "stanford_test_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_MODEL: Optional[ResNetPoseNet] = None
EVAL_RESULTS_DF: Optional[pd.DataFrame] = None

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def load_model_if_needed() -> ResNetPoseNet:
    global GLOBAL_MODEL
    if GLOBAL_MODEL is not None:
        return GLOBAL_MODEL

    if not CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found at {CHECKPOINT_PATH}"
        )

    try:
        model = ResNetPoseNet(pretrained=False)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        GLOBAL_MODEL = model
        return GLOBAL_MODEL
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Failed to load model checkpoint"
        )


def load_eval_results_df() -> Optional[pd.DataFrame]:
    global EVAL_RESULTS_DF
    if EVAL_RESULTS_DF is not None:
        return EVAL_RESULTS_DF

    if EVAL_RESULTS_PATH.exists():
        try:
            EVAL_RESULTS_DF = pd.read_csv(EVAL_RESULTS_PATH)
            # Create a normalized image filename column for fast lookup
            EVAL_RESULTS_DF["filename_only"] = EVAL_RESULTS_DF["image"].astype(str).apply(
                lambda p: Path(p).name
            )
        except Exception:
            EVAL_RESULTS_DF = None
    return EVAL_RESULTS_DF


@app.on_event("startup")
async def startup_event():
    try:
        load_model_if_needed()
    except Exception:
        pass  # Will be surfaced on demand by endpoints
    load_eval_results_df()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def get_health() -> Dict[str, str]:
    return {
        "status": "ok",
        "project": "NaVIP",
        "dataset": "Stanford 2D-3D-S Area 3",
    }


@app.get("/api/model/status")
def get_model_status() -> Dict[str, Any]:
    try:
        model = load_model_if_needed()
        return {
            "status": "available",
            "model": "ResNet-18 PoseNet",
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT_DIR)),
            "device": str(DEVICE),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Model checkpoint unavailable")


@app.post("/api/pose/predict")
async def predict_pose(image: UploadFile = File(...)) -> Dict[str, Any]:
    model = load_model_if_needed()

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided")

    try:
        input_tensor = IMAGE_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pos_tensor, orient_tensor = model(input_tensor)
            orient_tensor = orient_tensor / (
                torch.norm(orient_tensor, dim=1, keepdim=True) + 1e-8
            )

        pos = pos_tensor[0].cpu().tolist()
        orient = orient_tensor[0].cpu().tolist()
    except Exception:
        raise HTTPException(status_code=500, detail="Error performing model inference")

    response: Dict[str, Any] = {
        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "orientation": {
            "qw": float(orient[0]),
            "qx": float(orient[1]),
            "qy": float(orient[2]),
            "qz": float(orient[3]),
        },
    }

    # Match ground truth if filename exists in evaluation dataset
    if image.filename:
        df = load_eval_results_df()
        if df is not None:
            uploaded_filename = Path(image.filename).name
            matches = df[df["filename_only"] == uploaded_filename]
            if not matches.empty:
                row = matches.iloc[0]
                response["groundTruthPosition"] = {
                    "x": float(row["gt_x"]),
                    "y": float(row["gt_y"]),
                    "z": float(row["gt_z"]),
                }
                response["groundTruthOrientation"] = {
                    "qw": float(row["gt_qw"]),
                    "qx": float(row["gt_qx"]),
                    "qy": float(row["gt_qy"]),
                    "qz": float(row["gt_qz"]),
                }
                response["positionError"] = float(row["position_error_m"])
                response["orientationError"] = float(row["orientation_error_deg"])

    return response


def parse_rooms_data() -> List[Dict[str, Any]]:
    if not MAP_ROOMS_PATH.exists():
        raise HTTPException(status_code=503, detail="Room data unavailable")

    try:
        with open(MAP_ROOMS_PATH, "r") as f:
            raw_rooms = json.load(f)

        rooms: List[Dict[str, Any]] = []
        for key, data in raw_rooms.items():
            bounds_raw = data.get("bounds", [0, 0, 0, 0])
            min_x, max_x, min_y, max_y = (
                float(bounds_raw[0]),
                float(bounds_raw[1]),
                float(bounds_raw[2]),
                float(bounds_raw[3]),
            )
            # Format room type from name (e.g., lounge_2 -> lounge)
            room_type = key.split("_")[0] if "_" in key else key

            room_obj = {
                "id": key,
                "name": data.get("room_name", key).replace("_", " ").title(),
                "type": room_type,
                "availableForLocalization": True,
                "bounds": {
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                },
            }
            rooms.append(room_obj)
        return rooms
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Failed to parse room data")


@app.get("/api/rooms")
def get_rooms() -> List[Dict[str, Any]]:
    return parse_rooms_data()


@app.get("/api/rooms/{id}")
def get_room_by_id(id: string if False else str) -> Dict[str, Any]:
    rooms = parse_rooms_data()
    for room in rooms:
        if room["id"] == id or room["id"].lower() == id.lower():
            return room
    raise HTTPException(status_code=404, detail=f"Room with ID '{id}' not found")


@app.get("/api/map/area3/rooms")
def get_area3_rooms() -> List[Dict[str, Any]]:
    return parse_rooms_data()


@app.get("/api/map/area3/points")
def get_area3_points() -> List[Dict[str, Any]]:
    if not MAP_POINTS_PATH.exists():
        raise HTTPException(status_code=503, detail="Map points data unavailable")

    try:
        df = pd.read_csv(MAP_POINTS_PATH)
        points: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            points.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "type": str(row.get("room", "point")),
            })
        return points
    except Exception:
        raise HTTPException(status_code=503, detail="Failed to load map points")


@app.get("/api/map/area3")
def get_area3_map() -> Dict[str, Any]:
    rooms = parse_rooms_data()
    points = get_area3_points()

    min_x = min([p["x"] for p in points] + [r["bounds"]["x"] for r in rooms]) if points else 0.0
    max_x = max([p["x"] for p in points] + [r["bounds"]["x"] + r["bounds"]["width"] for r in rooms]) if points else 0.0
    min_y = min([p["y"] for p in points] + [r["bounds"]["y"] for r in rooms]) if points else 0.0
    max_y = max([p["y"] for p in points] + [r["bounds"]["y"] + r["bounds"]["height"] for r in rooms]) if points else 0.0

    return {
        "id": "area3",
        "name": "Stanford 2D-3D-S Area 3",
        "description": "3D Floorplan & Trajectory map of Area 3",
        "points": points,
        "rooms": rooms,
        "bounds": {
            "minX": float(min_x),
            "maxX": float(max_x),
            "minY": float(min_y),
            "maxY": float(max_y),
        },
    }


@app.get("/api/evaluation/results")
def get_evaluation_results() -> Dict[str, Any]:
    if not EVAL_SUMMARY_PATH.exists():
        raise HTTPException(status_code=503, detail="Evaluation results unavailable")

    try:
        with open(EVAL_SUMMARY_PATH, "r") as f:
            data = json.load(f)

        return {
            "testSamples": int(data.get("num_test_images", 0)),
            "successfulPredictions": int(data.get("successful_predictions", 0)),
            "failedPredictions": int(data.get("failed_predictions", 0)),
            "meanPositionError": float(round(data.get("mean_position_error_m", 0.0), 4)),
            "medianPositionError": float(round(data.get("median_position_error_m", 0.0), 4)),
            "meanOrientationError": float(round(data.get("mean_orientation_error_deg", 0.0), 4)),
            "medianOrientationError": float(round(data.get("median_orientation_error_deg", 0.0), 4)),
        }
    except Exception:
        raise HTTPException(status_code=503, detail="Failed to load evaluation results")
