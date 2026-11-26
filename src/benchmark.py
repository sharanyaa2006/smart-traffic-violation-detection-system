"""
Phase 7 — CPU performance benchmark.

Measures actual FPS for each stage of the pipeline separately, so you can
document engineering tradeoffs in your report (e.g. "helmet crop check
costs us X FPS, here's why we still do it two-stage instead of full-frame").

Usage: python src/benchmark.py
"""

import os
import sys
import time

import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PERSON_CLASS_ID = 0
MOTORCYCLE_CLASS_ID = 3


def benchmark(video_path=None, max_frames=150):
    video_path = video_path or config.DEFAULT_VIDEO_PATH
    helmet_model = YOLO(config.HELMET_MODEL_PATH)
    stock_model = YOLO(config.STOCK_MODEL_NAME)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("No frames read — check video path.")
        return

    print(f"Benchmarking on {len(frames)} frames from {os.path.basename(video_path)}")
    print("=" * 50)

    # 1. Stock model alone (person + motorcycle detection)
    start = time.time()
    for f in frames:
        stock_model.predict(f, conf=config.CONFIDENCE_THRESHOLD, imgsz=config.INFERENCE_IMG_SIZE,
                             classes=[PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID], verbose=False)
    stock_time = time.time() - start
    print(f"Stock model only:          {len(frames)/stock_time:.2f} FPS  ({stock_time:.2f}s total)")

    # 2. Stock model + tracking
    start = time.time()
    for f in frames:
        stock_model.track(f, conf=config.CONFIDENCE_THRESHOLD, imgsz=config.INFERENCE_IMG_SIZE,
                           classes=[PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID], persist=True, verbose=False,
                           tracker="bytetrack.yaml")
    tracking_time = time.time() - start
    print(f"Stock model + tracking:    {len(frames)/tracking_time:.2f} FPS  ({tracking_time:.2f}s total)")

    # 3. Helmet model alone on full frame (for comparison against crop-based)
    start = time.time()
    for f in frames:
        helmet_model.predict(f, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
    helmet_full_time = time.time() - start
    print(f"Helmet model (full frame): {len(frames)/helmet_full_time:.2f} FPS  ({helmet_full_time:.2f}s total)")

    # 4. Helmet model on a small crop (simulating the two-stage approach)
    sample_crop = frames[0][0:200, 0:200]
    start = time.time()
    for f in frames:
        helmet_model.predict(sample_crop, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
    helmet_crop_time = time.time() - start
    print(f"Helmet model (200x200 crop): {len(frames)/helmet_crop_time:.2f} FPS  ({helmet_crop_time:.2f}s total)")

    print("=" * 50)
    print("Note: full pipeline FPS (tracking + per-motorcycle helmet crop checks)")
    print("will be roughly: 1 / (1/tracking_fps + num_motorcycles * 1/helmet_crop_fps)")
    print("Run src/gui.py or run_violations.py on your real video for the true combined FPS.")


if __name__ == "__main__":
    benchmark()