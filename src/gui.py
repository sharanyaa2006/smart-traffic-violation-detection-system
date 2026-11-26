"""
Phase 4 — Tkinter GUI.

Ties together everything built so far:
  - detection (helmet model + stock model)
  - tracking (persistent IDs via ByteTrack)
  - rules (helmet / triple-riding / wrong-way confirmation logic)
  - database (logs confirmed violations to MySQL)
  - snapshot capture (saves a frame image whenever a violation confirms)

Usage: python src/gui.py
  File > Open Video  -> pick a video file
  Start               -> begins live processing, boxes drawn on canvas
  Stop                -> stops processing (can Open a new video after)
"""

import os
import sys
import time
from datetime import datetime
from tkinter import Tk, Frame, Label, Button, filedialog, Canvas, NW, END, Text, Scrollbar, RIGHT, Y, LEFT, BOTH

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import rules
from src import database
from src.tracking import ViolationTracker
from src.analytics import AnalyticsWindow

MOTORCYCLE_CLASS_ID = 3
PERSON_CLASS_ID = 0


class SmartTrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Traffic Violation Detection System")
        self.root.geometry("1000x700")

        self.video_path = None
        self.cap = None
        self.running = False
        self.frame_w, self.frame_h = None, None
        self.frame_count = 0

        self.helmet_model = None
        self.stock_model = None
        self.tracker_state = ViolationTracker()
        self.session_start_time = None

        self._build_menu()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Release the video capture cleanly before the window actually closes."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.root.destroy()

    # ── UI setup ─────────────────────────────────────────────
    def _build_menu(self):
        from tkinter import Menu
        menubar = Menu(self.root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video", command=self.open_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_layout(self):
        top_frame = Frame(self.root)
        top_frame.pack(side="top", fill="x", padx=8, pady=6)

        self.status_label = Label(top_frame, text="No video loaded", anchor="w")
        self.status_label.pack(side=LEFT, padx=4)

        self.start_btn = Button(top_frame, text="Start", command=self.start_processing, state="disabled")
        self.start_btn.pack(side=LEFT, padx=4)

        self.stop_btn = Button(top_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_btn.pack(side=LEFT, padx=4)

        self.analytics_btn = Button(top_frame, text="View Analytics", command=self.open_analytics)
        self.analytics_btn.pack(side=LEFT, padx=4)

        body_frame = Frame(self.root)
        body_frame.pack(side="top", fill=BOTH, expand=True, padx=8, pady=6)

        self.canvas = Canvas(body_frame, bg="black", width=760, height=520)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        log_frame = Frame(body_frame, width=220)
        log_frame.pack(side=LEFT, fill="y", padx=(8, 0))

        Label(log_frame, text="Violation Log", anchor="w").pack(fill="x")
        scrollbar = Scrollbar(log_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text = Text(log_frame, width=30, yscrollcommand=scrollbar.set)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

    # ── Video handling ───────────────────────────────────────
    def open_video(self):
        if self.running:
            self.stop_processing()

        path = filedialog.askopenfilename(
            initialdir=config.VIDEOS_DIR,
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
        )
        if not path:
            return

        self.video_path = path
        self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
        self.start_btn.config(state="normal")

        if self.helmet_model is None:
            self.status_label.config(text="Loading models...")
            self.root.update()
            self.helmet_model = YOLO(config.HELMET_MODEL_PATH)
            self.stock_model = YOLO(config.STOCK_MODEL_NAME)
            self.status_label.config(text=f"Loaded: {os.path.basename(path)}")

    def start_processing(self):
        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.log(f"ERROR: could not open {self.video_path}")
            return

        self.tracker_state.reset()
        self.frame_count = 0
        self.frame_w, self.frame_h = None, None
        self.running = True
        self.session_start_time = datetime.now()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self._process_next_frame()

    def stop_processing(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Stopped")

    def open_analytics(self):
        AnalyticsWindow(self.root, session_start=self.session_start_time)

    # ── Core per-frame loop ──────────────────────────────────
    def _process_next_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.log("Video ended.")
            self.stop_processing()
            return

        if self.frame_w is None:
            self.frame_h, self.frame_w = frame.shape[:2]

        self.frame_count += 1

        if self.frame_count % 200 == 0:
            self.tracker_state.prune(self.frame_count)

        results = self.stock_model.track(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            classes=list(config.VEHICLE_CLASSES_OF_INTEREST.keys()),
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        annotated = frame.copy()
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            cls_ids = boxes.cls.int().tolist()
            xyxy_list = boxes.xyxy.tolist()

            person_boxes = [b for c, b in zip(cls_ids, xyxy_list) if c == PERSON_CLASS_ID]
            motorcycle_entries = [(t, b) for t, c, b in zip(track_ids, cls_ids, xyxy_list) if c == MOTORCYCLE_CLASS_ID]

            for cls_id, box, tid in zip(cls_ids, xyxy_list, track_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = config.VEHICLE_CLASSES_OF_INTEREST.get(cls_id, "object")
                label = f"{class_name} #{tid}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), config.BOX_THICKNESS)
                cv2.putText(annotated, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            config.LABEL_FONT_SCALE, (0, 200, 0), config.LABEL_FONT_THICKNESS)

            for tid, m_box in motorcycle_entries:
                self.tracker_state.touch(tid, self.frame_count)
                self._check_violations(frame, annotated, tid, m_box, person_boxes)

        self._render_frame(annotated)
        self.root.after(1, self._process_next_frame)

    def _check_violations(self, raw_frame, annotated, tid, m_box, person_boxes):
        mx1, my1, mx2, my2 = m_box
        width_pad = (mx2 - mx1) * 0.15
        rider_check_box = (mx1 - width_pad, my1 - (my2 - my1) * 1.6, mx2 + width_pad, my2)

        rider_count = rules.count_riders_on_motorcycle(rider_check_box, person_boxes)
        no_helmet, helmet_confidence = self._get_helmet_status(raw_frame, m_box)
        center = rules.box_center(m_box)

        helmet_confirmed = self.tracker_state.update_helmet(tid, no_helmet)
        triple_confirmed = self.tracker_state.update_triple_riding(tid, rider_count)
        wrong_way_confirmed = self.tracker_state.update_wrong_way(tid, center)

        violation_labels = []
        if helmet_confirmed:
            violation_labels.append("NO HELMET")
        if triple_confirmed:
            violation_labels.append("TRIPLE RIDING")
        if wrong_way_confirmed:
            violation_labels.append("WRONG WAY")

        x1, y1, x2, y2 = map(int, m_box)

        for v_type in violation_labels:
            if self.tracker_state.is_duplicate(v_type, center, self.frame_count):
                continue  # likely the same vehicle under a new track_id, skip re-logging
            self.tracker_state.record_confirmation(v_type, center, self.frame_count)
            # only the helmet rule has a real model confidence score;
            # triple-riding and wrong-way are geometry/direction rules, not model detections
            conf_to_log = helmet_confidence if v_type == "NO HELMET" else None
            self._confirm_violation(annotated, v_type, tid, (x1, y1, x2, y2), confidence=conf_to_log)

        if violation_labels:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, " + ".join(violation_labels), (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    def _get_helmet_status(self, frame, motorcycle_box, padding=20):
        x1, y1, x2, y2 = motorcycle_box
        cx1 = max(0, int(x1 - padding))
        cy1 = max(0, int(y1 - padding * 4))
        cx2 = min(self.frame_w, int(x2 + padding))
        cy2 = min(self.frame_h, int(y2 + padding))

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return False, None

        results = self.helmet_model.predict(crop, conf=config.HELMET_CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return False, None

        names = results[0].names
        for box in boxes:
            if "without" in names[int(box.cls[0])].lower():
                return True, float(box.conf[0])
        return False, None

    def _confirm_violation(self, raw_frame, v_type, track_id, box, confidence=None):
        folder_name = v_type.replace(" ", "_")
        date_str = datetime.now().strftime("%Y-%m-%d")
        snapshot_folder = os.path.join(config.SNAPSHOTS_DIR, folder_name, date_str)
        os.makedirs(snapshot_folder, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot_filename = f"{folder_name}_{track_id}_{timestamp_str}.jpg"
        snapshot_path = os.path.join(snapshot_folder, snapshot_filename)
        cv2.imwrite(snapshot_path, raw_frame)

        violation_id = database.insert_violation(
            violation_type=v_type,
            track_id=track_id,
            confidence=confidence,
            video_source=os.path.basename(self.video_path) if self.video_path else None,
        )

        log_line = f"[Frame {self.frame_count}] Track #{track_id} — {v_type}"
        if violation_id:
            log_line += f" (logged, id={violation_id})"
        else:
            log_line += " (DB log failed — check MySQL connection)"
        self.log(log_line)

    # ── Display helpers ──────────────────────────────────────
    def _render_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        canvas_w = self.canvas.winfo_width() or 760
        canvas_h = self.canvas.winfo_height() or 520
        img.thumbnail((canvas_w, canvas_h))

        self._current_imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=self._current_imgtk)

    def log(self, message):
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        print(message)


if __name__ == "__main__":
    root = Tk()
    app = SmartTrafficApp(root)
    root.mainloop()