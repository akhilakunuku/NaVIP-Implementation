"""
Single Test Image Pose Prediction & Map Localization Inspector Tool.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.predict_one_image import run_single_image_localization

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else None
    run_single_image_localization(image_query=query)