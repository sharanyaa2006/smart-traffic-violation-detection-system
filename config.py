"""
Central configuration for Smart Traffic Violation Detection System.
All paths, model names, and thresholds live here — nowhere else in the
codebase should hardcode a path or a magic number.
"""

import os

# ── Base paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")

# ── Model files ───────────────────────────────────────────────
HELMET_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
STOCK_MODEL_NAME = "yolo11n.pt"  # pretrained, auto-downloads via ultralytics on first use

# ── Default test video ────────────────────────────────────────
DEFAULT_VIDEO_PATH = os.path.join(VIDEOS_DIR, "CCTV Footage.mp4")

# ── Detection thresholds ──────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25           # for stock model (person/car/motorcycle/etc) — lower is fine, classes are distinct
HELMET_CONFIDENCE_THRESHOLD = 0.4      # dialed back from 0.55 (too strict, missed real detections) —
                                        # tune this: raise if false positives return, lower if it misses real cases

# When the tracker loses an object and reassigns a new track_id to the
# same physical vehicle (common with occlusion/dense traffic), a
# violation can get logged twice under two different IDs. This suppresses
# a repeat log of the SAME violation type if one was already confirmed
# very recently, close by in the frame.
DUPLICATE_SUPPRESSION_RADIUS_PX = 200
DUPLICATE_SUPPRESSION_FRAME_WINDOW = 100
IOU_THRESHOLD = 0.45                  # for our OWN overlap checks (e.g. person-on-motorcycle) — NOT YOLO's internal NMS
INFERENCE_IMG_SIZE = 960              # bumped from default 640 — helps detect smaller objects in wide CCTV shots, costs some FPS

# ── Violation confirmation thresholds ─────────────────────────
# A violation must persist across this many consecutive frames
# before being logged — avoids single-frame flicker/false positives.
HELMET_CONFIRM_FRAMES = 10
TRIPLE_RIDING_CONFIRM_FRAMES = 8
WRONG_WAY_CONFIRM_FRAMES = 12

# Instead of requiring EVERY frame in the window to agree (too strict —
# a single missed frame from motion blur or tracker jitter resets the
# whole streak), require this fraction of recent frames to agree.
HELMET_CONFIRM_RATIO = 0.7
TRIPLE_RIDING_CONFIRM_RATIO = 0.7

# COCO class IDs for all vehicle/person types we want to detect and display.
# Only motorcycles go through the violation rule checks (helmet, triple
# riding, wrong-way) — the rest are detected and shown, but not checked.
VEHICLE_CLASSES_OF_INTEREST = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
TRIPLE_RIDING_MIN_COUNT = 3          # number of person-boxes on one motorcycle to count as triple riding

# Expected traffic flow direction, as a unit vector (dx, dy) in image coordinates.
# (0, 1)  = normal flow is downward (increasing y) on screen
# (0, -1) = normal flow is upward
# Update this per-video based on which way traffic normally moves in your footage.
EXPECTED_FLOW_VECTOR = (0, 1)
WRONG_WAY_ANGLE_THRESHOLD_DEGREES = 145   # raised from 120 — stricter, only flags clearly opposite movement
TRACK_HISTORY_LENGTH = 40                  # raised from 30 — longer, more stable movement history before confirming

# ── MySQL database config ─────────────────────────────────────
# Credentials are loaded from a .env file (see .env.example) instead of
# being hardcoded here. Copy .env.example to .env and fill in your own
# values — .env is gitignored so your password never gets committed.
from dotenv import load_dotenv
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "traffic_violations"),
}

# ── Display settings ───────────────────────────────────────────
LABEL_FONT_SCALE = 0.35       # smaller label text (was implicitly 0.5) — less screen clutter
LABEL_FONT_THICKNESS = 1
BOX_THICKNESS = 1             # thinner box lines too, helps with dense scenes
# Placeholder — will be corrected once we print model.names from best.pt
HELMET_CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet",
}