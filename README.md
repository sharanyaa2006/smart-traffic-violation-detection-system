# Smart Traffic Violation Detection System

A computer vision pipeline that automatically detects traffic violations from CCTV footage. Built with YOLOv11 for object detection, ByteTrack for vehicle tracking, MySQL for logging, and a desktop Tkinter interface for reviewing incidents.

The system monitors two-wheelers and flags three specific violations:
1. **Riding without a helmet**
2. **Triple riding** (three or more people on a single motorcycle)
3. **Wrong-way riding** (traveling against the defined traffic direction)

When a violation is confirmed, the system saves an evidence snapshot to disk, records the incident details into a MySQL database, and displays it in real time on the desktop dashboard.

---

## How It Works

Detecting violations from a raw CCTV feed reliably requires more than just running an object detector on every frame. This system uses a multi-stage approach to minimize false positives and tracker noise:

1. **Vehicle & Person Detection:** YOLOv11 identifies persons, motorcycles, cars, buses, and trucks across the frame.
2. **Multi-Object Tracking:** ByteTrack assigns persistent tracking IDs to vehicles so they can be followed from frame to frame.
3. **Targeted Crop Classification:** For helmet checking, motorcycle regions are cropped and passed to a dedicated custom YOLOv11 classifier trained specifically on helmet and bare-head classes. Running the classifier on crops rather than the full wide-angle frame significantly improves detection accuracy on small objects.
4. **Geometric & Vector Analysis:**
   - **Triple Riding:** Measures spatial overlap (IoU) and counts the number of detected person bounding boxes associated with a single motorcycle.
   - **Wrong-Way Driving:** Compares the movement trajectory vector of a vehicle over a historical frame window against the expected lane direction vector.
5. **Temporal Streak Confirmation:** To prevent a single blurry frame or detector flicker from triggering a false ticket, a violation is only confirmed if it persists across at least **70% of frames** within a rolling observation window (e.g., 10 frames for helmets, 8 for triple riding, 12 for wrong-way).
6. **Spatial Duplicate Suppression:** If tracking briefly drops and reassigns an ID to the same physical vehicle, a spatial radius filter suppresses repeat logging of the same violation within a short time window.

---

## Repository Structure

```text
smart_traffic/
├── main.py                     # Entry point for the desktop GUI application
├── config.py                   # Central settings, model paths, and detection thresholds
├── requirements.txt            # Python dependencies
├── reorganize_snapshots.py     # Utility script to sort snapshots by type and date
│
├── src/
│   ├── detection.py            # YOLO model loading and bounding-box inference
│   ├── tracking.py             # ByteTrack integration and trajectory history
│   ├── rules.py                # Violation decision logic (helmet, rider count, trajectory)
│   ├── run_violations.py       # Standalone CLI script to process videos without the GUI
│   ├── database.py             # MySQL connector and violation insertion queries
│   ├── gui.py                  # Tkinter desktop application with live stream playback
│   ├── analytics.py            # Summary dashboard displaying charts and historical logs
│   ├── evaluate_model.py       # Evaluates trained models against validation/test sets
│   └── benchmark.py            # Measures CPU/GPU inference latency and FPS
│
├── models/
│   └── best.pt                 # Custom-trained YOLOv11 weights for helmet detection
│
├── database/
│   └── schema.sql              # Database table definitions for violations
│
├── docs/
│   └── metrics_report.md       # Benchmarking notes, mAP figures, and performance analysis
│
├── notebooks/
│   └── model_evaluation.ipynb  # Interactive evaluation notebook
│
└── tests/
    ├── test_rules.py           # Unit tests for violation detection logic
    └── test_tracking.py        # Unit tests for tracking and trajectory calculations
```

---

## Prerequisites

- **Python:** 3.8 to 3.11
- **MySQL Server:** 5.7 or 8.0+
- **Hardware:** Standard CPU (tested on Intel Core i7) or an NVIDIA GPU for higher FPS.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/sharanyaa2006/smart-traffic-violation-detection-system.git
cd smart-traffic-violation-detection-system
```

### 2. Set up a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up the database
Import the database schema into your local MySQL server:
```bash
mysql -u root -p < database/schema.sql
```

Create a `.env` file in the project root by copying the template:
```bash
cp .env.example .env
```

Open `.env` and enter your database credentials:
```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=traffic_violations
```

---

## Running the Application

### Desktop GUI
The desktop interface lets you select a video, view live tracking bounding boxes, and inspect confirmed violation alerts along with snapshots.
```bash
python main.py
```

### Headless CLI Pipeline
If you want to process a video file without opening the user interface:
```bash
python src/run_violations.py
```

### Running Tests
Unit tests cover the core geometry, trajectory calculation, and threshold confirmation rules:
```bash
python -m pytest tests/ -v
```

### Performance Benchmark
To measure frame rate and latency on your specific machine:
```bash
python src/benchmark.py
```

---

## Configuration & Tuning

All adjustable thresholds live in [`config.py`](config.py). You can tune these values depending on camera angles, scene density, and road layout:

- `CONFIDENCE_THRESHOLD`: Confidence cutoff for general YOLO vehicle/person detection (default: `0.25`).
- `HELMET_CONFIDENCE_THRESHOLD`: Confidence threshold for the helmet classification model (default: `0.40`).
- `EXPECTED_FLOW_VECTOR`: Unit vector `(dx, dy)` representing normal traffic direction in image coordinates. Default is `(0, 1)` for downward traffic flow on screen. Update this to match your camera perspective.
- `WRONG_WAY_ANGLE_THRESHOLD_DEGREES`: Angular deviation threshold (in degrees) from the expected flow vector required to flag wrong-way travel (default: `145°`).
- `HELMET_CONFIRM_FRAMES` / `HELMET_CONFIRM_RATIO`: Number of consecutive frames and ratio required before confirming an unhelmeted rider (default: 10 frames with at least 70% agreement).
- `DUPLICATE_SUPPRESSION_RADIUS_PX`: Radius in pixels to prevent re-logging the same vehicle if its track ID changes due to brief occlusion (default: `200px`).

---

## Model Evaluation & Performance

Detailed test metrics and benchmark findings are documented in [`docs/metrics_report.md`](docs/metrics_report.md).

- **Helmet Detection:** Evaluated on a test split of 151 images (409 labeled instances).
  - Precision: `93.4%`
  - Recall: `81.8%` (`89.3%` for With Helmet, `74.2%` for Without Helmet)
  - mAP@50: `90.3%`
- **Speed:** On a standard laptop CPU (Intel Core i7-1355U), the complete pipeline runs at **5–10 FPS**. This is suitable for recorded CCTV video inspection and offline batch processing. Real-time 30 FPS monitoring requires a CUDA-capable GPU.

---

## Known Limitations

- **Road Geometry:** Wrong-way detection expects straight road sections with predictable flow. Intersections and roundabouts require per-zone directional masks.
- **Occlusions in Crowds:** Triple riding uses bounding-box intersection heuristics. In dense pedestrian crowds adjacent to stopped bikes, false overlaps can occasionally occur.
- **Dataset Diversity:** Recall on riders without helmets is currently 74%, meaning difficult angles, low lighting, or unique hairstyles may occasionally be missed. Adding more varied non-helmet training data is the planned next step.

---

## License

This project is licensed under the [MIT License](LICENSE).
