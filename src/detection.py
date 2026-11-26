"""
Phase 1 — Detection sanity check.

Loads the custom-trained helmet model (best.pt) and runs it on a test
video to confirm:
  1. The model loads correctly
  2. What classes it actually outputs (model.names) — critical, since
     the violation rules later depend on knowing the exact class indices
  3. Bounding boxes draw correctly on live video via cv2.imshow()
  4. Rough FPS on CPU, to set expectations before adding tracking + rules

Run this file directly to sanity-check before moving to Phase 2.
"""

import os
import sys
import time

import cv2
from ultralytics import YOLO

# allow running this file directly (python src/detection.py) by adding
# the project root to sys.path so `import config` works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_helmet_model():
    """Load the custom-trained helmet/no-helmet YOLO model."""
    if not os.path.exists(config.HELMET_MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find model at {config.HELMET_MODEL_PATH}. "
            "Make sure best.pt is inside the models/ folder."
        )
    model = YOLO(config.HELMET_MODEL_PATH)
    return model


def load_stock_model():
    """
    Load the pretrained stock YOLO11 model (COCO classes).
    Used for person / motorcycle / vehicle detection — no training needed,
    these classes already exist in COCO.
    Auto-downloads the weights on first run if not already cached.
    """
    model = YOLO(config.STOCK_MODEL_NAME)
    return model


# COCO class indices we actually care about from the stock model.
# (Full COCO has 80 classes — we only need these for our rules and display.)
STOCK_CLASSES_OF_INTEREST = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def run_sanity_check(video_path=None, show_window=True, max_frames=None):
    """
    Run the helmet model on a video and report:
      - class names the model knows
      - average FPS
      - (optionally) show live annotated frames
    """
    video_path = video_path or config.DEFAULT_VIDEO_PATH
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Could not find video at {video_path}")

    model = load_helmet_model()

    print("=" * 50)
    print("Model loaded successfully.")
    print(f"Model classes (model.names): {model.names}")
    print("=" * 50)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_count = 0
    total_inference_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model.predict(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        elapsed = time.time() - start
        total_inference_time += elapsed
        frame_count += 1

        annotated_frame = results[0].plot()  # draws boxes + labels automatically

        if show_window:
            cv2.imshow("Phase 1 Sanity Check - press q to quit", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if frame_count > 0:
        avg_fps = frame_count / total_inference_time
        print("=" * 50)
        print(f"Processed {frame_count} frames.")
        print(f"Average inference FPS (CPU): {avg_fps:.2f}")
        print("=" * 50)
    else:
        print("No frames were processed — check the video file.")


def draw_boxes(frame, results, color, label_prefix=""):
    """
    Manually draw bounding boxes + labels on a frame from a YOLO results object.
    Doing this manually (instead of relying on results.plot() chaining between
    two models) avoids version-dependent bugs where .plot(img=...) silently
    fails to overlay onto an already-annotated frame.
    """
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame

    names = results[0].names

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{label_prefix}{names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.LABEL_FONT_SCALE, config.LABEL_FONT_THICKNESS
        )
        cv2.rectangle(
            frame, (x1, y1 - text_h - 4), (x1 + text_w + 3, y1), color, -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.LABEL_FONT_SCALE,
            (255, 255, 255),
            config.LABEL_FONT_THICKNESS,
        )

    return frame


def run_combined_check(video_path=None, show_window=True, max_frames=None):
    """
    Phase 2 — run BOTH models on the same video:
      - stock YOLO11 for person / car / motorcycle / bus / truck
      - custom helmet model for helmet / no-helmet
    Draws both sets of boxes on the same frame manually (see draw_boxes)
    and reports combined FPS.
    """
    video_path = video_path or config.DEFAULT_VIDEO_PATH
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Could not find video at {video_path}")

    helmet_model = load_helmet_model()
    stock_model = load_stock_model()

    print("=" * 50)
    print(f"Helmet model classes: {helmet_model.names}")
    print(f"Stock model classes of interest: {STOCK_CLASSES_OF_INTEREST}")
    print("=" * 50)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_count = 0
    total_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        helmet_results = helmet_model.predict(
            frame,
            conf=config.HELMET_CONFIDENCE_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            verbose=False,
        )
        stock_results = stock_model.predict(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            classes=list(STOCK_CLASSES_OF_INTEREST.keys()),
            verbose=False,
        )

        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1

        # Draw manually: helmet boxes in red, stock (vehicle/person) boxes in green
        annotated = frame.copy()
        annotated = draw_boxes(annotated, stock_results, color=(0, 200, 0))
        annotated = draw_boxes(annotated, helmet_results, color=(0, 0, 255))

        if show_window:
            cv2.imshow("Phase 2 Combined Check - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if frame_count > 0:
        avg_fps = frame_count / total_time
        print("=" * 50)
        print(f"Processed {frame_count} frames with BOTH models.")
        print(f"Average combined FPS (CPU): {avg_fps:.2f}")
        print("=" * 50)


def run_two_stage_check(video_path=None, show_window=True, max_frames=None, crop_padding=20):
    """
    Phase 2 (fixed) — two-stage detection to handle scale mismatch:

      Stage 1: stock model detects person / car / motorcycle / bus / truck
               on the FULL frame (they're roughly the right scale already)

      Stage 2: for each motorcycle box, CROP that region out of the frame
               with some padding, then run the helmet model on the crop
               instead of the full frame. This matches the close-up scale
               the helmet model was likely trained on, fixing the "no
               helmet boxes at all" issue that happens when a rider's head
               is a tiny fraction of a wide CCTV frame.

    Helmet detections are mapped back to full-frame coordinates before
    drawing, so boxes still line up correctly on the original video.
    """
    video_path = video_path or config.DEFAULT_VIDEO_PATH
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Could not find video at {video_path}")

    helmet_model = load_helmet_model()
    stock_model = load_stock_model()

    print("=" * 50)
    print(f"Helmet model classes: {helmet_model.names}")
    print(f"Stock model classes of interest: {STOCK_CLASSES_OF_INTEREST}")
    print("Running TWO-STAGE detection (crop motorcycle regions for helmet check)")
    print("=" * 50)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_count = 0
    total_time = 0.0
    frame_h, frame_w = None, None

    MOTORCYCLE_CLASS_ID = 3  # matches STOCK_CLASSES_OF_INTEREST

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_h is None:
            frame_h, frame_w = frame.shape[:2]

        start = time.time()

        stock_results = stock_model.predict(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            classes=list(STOCK_CLASSES_OF_INTEREST.keys()),
            verbose=False,
        )

        annotated = frame.copy()
        annotated = draw_boxes(annotated, stock_results, color=(0, 200, 0))

        boxes = stock_results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != MOTORCYCLE_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cx1 = max(0, x1 - crop_padding)
                cy1 = max(0, y1 - crop_padding * 4)  # extra upward padding for the head
                cx2 = min(frame_w, x2 + crop_padding)
                cy2 = min(frame_h, y2 + crop_padding)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                helmet_results = helmet_model.predict(
                    crop,
                    conf=config.HELMET_CONFIDENCE_THRESHOLD,
                    verbose=False,
                )

                helmet_boxes = helmet_results[0].boxes
                if helmet_boxes is None or len(helmet_boxes) == 0:
                    continue

                helmet_names = helmet_results[0].names
                for hbox in helmet_boxes:
                    hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0].tolist())
                    hconf = float(hbox.conf[0])
                    hcls = int(hbox.cls[0])
                    hlabel = f"{helmet_names[hcls]} {hconf:.2f}"

                    fx1, fy1 = hx1 + cx1, hy1 + cy1
                    fx2, fy2 = hx2 + cx1, hy2 + cy1

                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), config.BOX_THICKNESS)
                    cv2.putText(
                        annotated,
                        hlabel,
                        (fx1 + 1, fy1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        config.LABEL_FONT_SCALE,
                        (0, 0, 255),
                        config.LABEL_FONT_THICKNESS,
                    )

        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1

        if show_window:
            cv2.imshow("Phase 2 Two-Stage Check - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if frame_count > 0:
        avg_fps = frame_count / total_time
        print("=" * 50)
        print(f"Processed {frame_count} frames with two-stage detection.")
        print(f"Average FPS (CPU): {avg_fps:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    # Phase 1: run_sanity_check()
    # Phase 2 (full-frame, has scale mismatch issue): run_combined_check()
    # Phase 2 (fixed, two-stage crop-based helmet check): run_two_stage_check()
    run_two_stage_check()