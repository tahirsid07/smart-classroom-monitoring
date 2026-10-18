# Smart Classroom Monitoring System

A Python-based classroom monitoring MVP. It reads a webcam or video, detects supported activities with YOLO, stores incident metadata in SQLite, and can send optional Telegram alerts.

## Project layout

```text
smart-classroom-monitoring/
├── backend/
│   ├── classroom_monitor/   # Python application modules
│   ├── tests/               # Existing automated tests
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/                # Reserved for a future dashboard or web UI
├── data/                    # Local-only runtime records and student gallery
├── .env.example
└── README.md
```

The current project is a command-line/OpenCV application and does not yet include a separate web frontend.

## Features

- Webcam and video-file monitoring
- YOLO detections mapped to mobile-phone use and eating labels
- Confidence filtering and duplicate-alert cooldown
- Incident metadata stored in SQLite
- Optional Telegram notifications
- Optional face matching against a locally maintained student gallery

A standard COCO-trained YOLO model can detect cell phones, but it does not reliably classify eating. Use a separately trained model for that activity. Alerts should be reviewed by a human; detections alone are not proof of misconduct.

## Setup (macOS/Linux)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install -e ./backend
cp .env.example .env
```

Edit `.env` if needed, then load its values into the current terminal before launching:

```bash
set -a
source .env
set +a
```

Keep Telegram tokens and chat IDs private. `.env` is excluded from Git.

## Run

Start the webcam monitor:

```bash
classroom-monitor
```

Choose a camera or video and model explicitly:

```bash
classroom-monitor --source 0 --model yolo11n.pt
classroom-monitor --source path/to/classroom.mp4 --model path/to/best.pt
```

On a headless machine, add `--no-display`. Press `q` in the OpenCV window to stop. The YOLO weights can be downloaded by Ultralytics when the named model is first used.

## Student gallery

For local face matching, add consented images under `data/students/<student-id>/` and an optional `details.json` file. Example:

```text
data/students/
└── STU-001/
    ├── face.jpg
    └── details.json
```

```json
{"name": "Student Name", "student_id": "STU-001", "grade": "10-A"}
```

The app downloads its OpenCV face models into `data/models/` on first display-enabled start. Student images, captures, model weights, and the SQLite database are local runtime files and are intentionally excluded from GitHub and the source ZIP. Do not add identifiable student data to version control.

## Tests

```bash
python -m pytest backend/tests -q
```

## Privacy

Get institutional approval and inform students before recording. Limit access to images and the database, set a retention period, protect Telegram credentials, and review alerts before taking action.
