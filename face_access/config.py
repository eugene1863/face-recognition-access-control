"""Central configuration for the face access-control system."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_DIR = DATA_DIR / "model"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
DB_PATH = DATA_DIR / "access.db"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABELS_PATH = MODEL_DIR / "labels.json"

for d in (FACES_DIR, MODEL_DIR, SNAPSHOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Face detection
FACE_SIZE = (200, 200)          # size faces are normalized to before training/recognition
MIN_FACE_SIZE = (80, 80)        # ignore detections smaller than this (reduces false positives)

# Enrollment
SAMPLES_PER_USER = 25
CAPTURE_INTERVAL_MS = 150       # gap between captured samples during enrollment

# Recognition
# LBPH confidence is a *distance* — lower means a closer match.
# Below this value we treat it as a positive identification.
CONFIDENCE_THRESHOLD = 95.0
RECOGNITION_COOLDOWN_SEC = 5    # avoid re-logging the same person every frame

UNKNOWN_LABEL = "Unknown"
