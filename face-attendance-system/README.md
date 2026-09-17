# Face Recognition Attendance System

A basic, fully local, offline face-recognition attendance system built with
Python and OpenCV. No cloud services, no paid APIs, no `dlib` install
headaches — it uses OpenCV's built-in Haar Cascade (face detection) and
LBPH recognizer (face recognition).

## How it works

1. **`register_faces.py`** — Turns on your webcam, captures ~60 face
   photos of a person, and saves them in `dataset/`. Also records their
   name + ID in `students.csv`.
2. **`train_model.py`** — Trains a recognizer on everyone currently in
   `dataset/` and saves the trained model to `trainer/trainer.yml`.
3. **`attendance_system.py`** — Runs the live webcam feed, recognizes
   registered faces, and logs attendance (once per person per day) to
   `Attendance/Attendance_YYYY-MM-DD.csv`.

## 1. Prerequisites

- Python 3.9–3.11 installed (Python 3.12 also usually works).
- A working webcam.
- Windows, macOS, or Linux — no OS-specific steps needed.

Check your Python version:
```bash
python --version
```

## 2. Setup

Unzip the project, then open a terminal inside the `face-attendance-system`
folder.

**Create a virtual environment (recommended):**
```bash
python -m venv venv
```

Activate it:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

**Install dependencies:**
```bash
pip install -r requirements.txt
```

If `opencv-contrib-python` fails to install, upgrade pip first:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Usage

### Step A — Register each person
```bash
python register_faces.py
```
Enter their name when prompted, then look at the camera and hold still
while it captures ~60 images (a few seconds). Repeat this once **per
person** you want the system to recognize.

### Step B — Train the model
```bash
python train_model.py
```
Run this after registering everyone (or any time you add a new person).

### Step C — Run attendance
```bash
python attendance_system.py
```
A window opens showing your webcam feed with a green box + name around
recognized faces, and a red "Unknown" box around unrecognized ones. Each
person is marked present **once** per day. Press `q` to quit.

Attendance logs are saved to `Attendance/Attendance_2026-09-13.csv` (one
file per day) with columns: `ID, Name, Time`.

## 4. Project structure

```
face-attendance-system/
├── register_faces.py      # Step 1: capture face samples
├── train_model.py         # Step 2: train the recognizer
├── attendance_system.py   # Step 3: run live recognition + attendance
├── requirements.txt
├── students.csv           # auto-created: ID -> Name mapping
├── dataset/                # auto-created: captured face images
├── trainer/
│   └── trainer.yml        # auto-created: trained model
└── Attendance/
    └── Attendance_YYYY-MM-DD.csv   # auto-created: daily logs
```

## 5. Anti-spoofing (liveness detection)

To stop someone from marking attendance using a photo or a static image on
a phone/tablet, `attendance_system.py` now requires a **liveness check**
(`liveness.py`) before it marks anyone present:

- **Blink detection** — the person must show a real open → closed → open
  eye cycle. A printed photo or a still image can't blink, so it never
  passes.
- **Flatness/texture check** — printed photos and screens tend to look
  "flatter" than a real face under normal lighting; unusually flat faces
  are rejected.
- **6-second rolling challenge** — if liveness isn't proven in time, it
  resets and keeps asking, so someone can't just wait it out.

While recognized, a face shows an orange box with a live status like
`"Please blink (0/1) - 4s"` until it turns green as `"Marked present"`.

**Be realistic about what this does and doesn't stop.** This is a
classical computer-vision check, not commercial-grade anti-spoofing (the
kind that uses infrared/depth cameras or trained deep-learning models). It
reliably blocks:
- a printed photo held up to the camera
- a static image displayed on a phone/tablet screen

It is **not guaranteed** to stop a determined attacker holding up a
**video** of the real person's face blinking naturally — that's a genuinely
hard problem, and no basic webcam-only system can fully solve it. If you
need attendance integrity for something high-stakes (exams, payroll,
security access), this project should be paired with human spot-checks,
not relied on alone.

Tuning knobs, all at the top of `attendance_system.py`:
- `REQUIRED_BLINKS` — raise to 2 for a stricter check.
- `CHALLENGE_TIMEOUT_SECONDS` — how long a person has to blink before the
  challenge resets.
- `FLATNESS_THRESHOLD` — lower this if real faces are being rejected as
  "flat" (e.g. in low light); raise it if photos are still getting through.

## 6. Tuning & troubleshooting

- **Webcam won't open**: try changing `cv2.VideoCapture(0)` to `1` or `2`
  in the scripts (some laptops have multiple camera indices).
- **Too many "Unknown" results**: raise `CONFIDENCE_THRESHOLD` in
  `attendance_system.py` (e.g. 65 → 80).
- **False matches between different people**: lower
  `CONFIDENCE_THRESHOLD` (e.g. 65 → 50), and re-register with more/better
  lit samples.
- **Poor accuracy overall**: capture samples in good, even lighting, with
  the face filling a reasonable portion of the frame, and from slightly
  different angles.
- **Adding a new person later**: just run `register_faces.py` again for
  them, then re-run `train_model.py` — it retrains on everyone in
  `dataset/`, so you don't need to redo existing people.

## 7. Limitations (this is a "basic" version)

- LBPH is lightweight and works well for small groups (a classroom/small
  office), but is less accurate than deep-learning face embeddings for
  large populations (hundreds of people).
- Liveness detection (see section 5) blocks photo/screen spoofing but not
  a sophisticated video replay attack.
- No GUI/dashboard — attendance is a CSV file. This is a solid base to
  add a dashboard (e.g. with Streamlit or a Flask web app) on top of
  later.
