"""
Phase 3 — Full pipeline test: tracking + all 3 violation rules together.

For each frame:
  1. Stock model tracks person / motorcycle / car / bus / truck (persistent
     IDs via ByteTrack, built into ultralytics' model.track())
  2. For each motorcycle track:
       - crop + run helmet model (two-stage, same fix as Phase 2)
       - count overlapping person boxes -> triple-riding check
       - track center position over time -> wrong-way check
  3. ViolationTracker confirms violations only after they persist across
     several frames (see src/tracking.py), and only reports each once.
  4. Confirmed violations are printed to console and boxes turn RED.

This is a console + live-window test — MySQL logging and the Tkinter GUI
get wired in during Phase 4/5.
"""

import os
import sys
import time

import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import rules
from tracking import ViolationTracker

MOTORCYCLE_CLASS_ID = 3
PERSON_CLASS_ID = 0


def load_models():
    helmet_model = YOLO(config.HELMET_MODEL_PATH)
    stock_model = YOLO(config.STOCK_MODEL_NAME)
    return helmet_model, stock_model


def get_helmet_status_for_crop(helmet_model, frame, motorcycle_box, frame_w, frame_h, padding=20):
    """
    Two-stage helmet check (same fix as Phase 2): crop around the
    motorcycle box with extra upward padding for the head, run the helmet
    model on the crop only. Returns (is_no_helmet, confidence) — confidence
    is None when no no-helmet detection is found in the crop.
    """
    x1, y1, x2, y2 = motorcycle_box
    cx1 = max(0, int(x1 - padding))
    cy1 = max(0, int(y1 - padding * 4))
    cx2 = min(frame_w, int(x2 + padding))
    cy2 = min(frame_h, int(y2 + padding))

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False, None

    results = helmet_model.predict(crop, conf=config.HELMET_CONFIDENCE_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False, None

    names = results[0].names
    # Robust check: look for "without" in the label (case-insensitive)
    # instead of an exact string match — survives label text changing
    # slightly between retrains (e.g. "no-helmet" vs "Without Helmet").
    for box in boxes:
        cls_id = int(box.cls[0])
        if "without" in names[cls_id].lower():
            return True, float(box.conf[0])
    return False, None


def run(video_path=None, show_window=True, max_frames=None):
    video_path = video_path or config.DEFAULT_VIDEO_PATH
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Could not find video at {video_path}")

    helmet_model, stock_model = load_models()
    tracker_state = ViolationTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_w, frame_h = None, None
    frame_count = 0
    confirmed_violations_log = []  # list of dicts, used later for MySQL insert (Phase 5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_w is None:
            frame_h, frame_w = frame.shape[:2]

        frame_count += 1

        if frame_count % 200 == 0:
            tracker_state.prune(frame_count)

        # model.track() with persist=True keeps the same tracker state
        # across calls, so track_ids stay consistent frame to frame.
        results = stock_model.track(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            classes=list(config.VEHICLE_CLASSES_OF_INTEREST.keys()),
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        boxes = results[0].boxes
        annotated = frame.copy()

        if boxes is None or boxes.id is None:
            # tracker hasn't assigned IDs yet (e.g. first frame) — just show raw frame
            if show_window:
                cv2.imshow("Phase 3 - Violations - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and frame_count >= max_frames:
                break
            continue

        track_ids = boxes.id.int().tolist()
        cls_ids = boxes.cls.int().tolist()
        xyxy_list = boxes.xyxy.tolist()

        # split into person boxes and motorcycle boxes (with their track_ids)
        person_boxes = []
        motorcycle_entries = []  # (track_id, box)
        for tid, cls_id, box in zip(track_ids, cls_ids, xyxy_list):
            if cls_id == PERSON_CLASS_ID:
                person_boxes.append(box)
            elif cls_id == MOTORCYCLE_CLASS_ID:
                motorcycle_entries.append((tid, box))

        # draw all raw boxes first (green = normal)
        for cls_id, box, tid in zip(cls_ids, xyxy_list, track_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = config.VEHICLE_CLASSES_OF_INTEREST.get(cls_id, "object")
            label = f"{class_name} #{tid}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), config.BOX_THICKNESS)
            cv2.putText(
                annotated, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                config.LABEL_FONT_SCALE, (0, 200, 0), config.LABEL_FONT_THICKNESS,
            )

        # per-motorcycle checks: helmet, triple-riding, wrong-way
        for tid, m_box in motorcycle_entries:
            tracker_state.touch(tid, frame_count)
            # Riders' person-boxes extend well above the motorcycle's own
            # box (torso/head), so checking overlap against the raw
            # motorcycle box misses riders almost every time. Extend the
            # box upward before counting riders on it.
            mx1, my1, mx2, my2 = m_box
            width_pad = (mx2 - mx1) * 0.15
            rider_check_box = (
                mx1 - width_pad,
                my1 - (my2 - my1) * 1.6,
                mx2 + width_pad,
                my2,
            )

            rider_count = rules.count_riders_on_motorcycle(rider_check_box, person_boxes)
            no_helmet, helmet_confidence = get_helmet_status_for_crop(helmet_model, frame, m_box, frame_w, frame_h)
            center = rules.box_center(m_box)

            helmet_confirmed = tracker_state.update_helmet(tid, no_helmet)
            triple_confirmed = tracker_state.update_triple_riding(tid, rider_count)
            wrong_way_confirmed = tracker_state.update_wrong_way(tid, center)

            x1, y1, x2, y2 = map(int, m_box)
            violation_labels = []

            if helmet_confirmed:
                violation_labels.append("NO HELMET")
            if triple_confirmed:
                violation_labels.append("TRIPLE RIDING")
            if wrong_way_confirmed:
                violation_labels.append("WRONG WAY")

            for v_type in violation_labels:
                if tracker_state.is_duplicate(v_type, center, frame_count):
                    continue  # likely the same vehicle under a new track_id, skip re-logging
                tracker_state.record_confirmation(v_type, center, frame_count)
                conf_to_log = helmet_confidence if v_type == "NO HELMET" else None
                entry = {
                    "type": v_type,
                    "track_id": tid,
                    "frame": frame_count,
                    "box": (x1, y1, x2, y2),
                    "confidence": conf_to_log,
                }
                confirmed_violations_log.append(entry)
                print(f"[VIOLATION] Frame {frame_count} | Track #{tid} | {v_type}")

            if violation_labels:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label_text = " + ".join(violation_labels)
                cv2.putText(
                    annotated, label_text, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                )

        if show_window:
            cv2.imshow("Phase 3 - Violations - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    print("=" * 50)
    print(f"Total confirmed violations: {len(confirmed_violations_log)}")
    for entry in confirmed_violations_log:
        print(entry)
    print("=" * 50)

    return confirmed_violations_log


if __name__ == "__main__":
    run()