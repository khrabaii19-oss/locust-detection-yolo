# Flea Detection & Laser Targeting System (YOLOv8)

Real-time detection and tracking of fleas using YOLOv8, integrated with a
mechanical laser targeting system. A vision processor (PC / Jetson) runs
YOLOv8 inference on a live camera feed and sends normalized target
coordinates to a Raspberry Pi, which drives the laser servos.

---

## 1. Project Overview

| Component        | Role                                                               |
|------------------|--------------------------------------------------------------------|
| Vision Processor | Runs YOLOv8, captures camera frames, computes target coordinates   |
| Raspberry Pi     | Receives coordinates over serial/UDP, drives laser pan/tilt servos |
| Laser Assembly   | Mechanical pan/tilt stage + laser diode aimed at the target        |

**Key design decisions:**

- **Normalized coordinates (0–1)** — the vision module never sends pixel
  values. `(0,0)` is the top-left, `(1,1)` the bottom-right of the frame,
  so `(0.5, 0.5)` is dead-center. This makes the system independent of
  camera resolution.
- **Closest-target selection** — among multiple fleas, the detection with
  the **largest bounding-box area** is treated as the nearest flea (same
  object class ⇒ area scales inversely with distance). Only that target is
  transmitted.
- **Low latency** — a dedicated capture thread decouples camera I/O from
  inference; inference runs at `640×640` with FP16 on GPU; transmission is
  rate-limited to avoid flooding the serial link.

## 2. System Architecture

```
                         VISION PROCESSOR (PC / Jetson)
   +-------------------------------------------------------------+
   |  USB Camera / IP Camera  --->  CameraStream (thread)         |
   |                                   |                         |
   |                                   v                         |
   |                        YOLOv8 inference (best.pt)           |
   |                                   |                         |
   |                         select closest flea (largest box)   |
   |                                   |                         |
   |                    normalize center -> (x, y) in [0,1]      |
   |                                   |                         |
   |                    ActuatorLink (serial / UDP)              |
   +-----------------------------------|-------------------------+
                                       |
                         JSON {"x":0.45,"y":0.30}\n   or   CSV 0.45,0.30\n
                                       |
                                       v
   +-------------------------------------------------------------+
   |  RASPBERRY PI                                                |
   |   serial/UDP listener -> parse -> PID servo controller       |
   |                                                              |
   |   Pan/Tilt servo driver  --->  Laser diode assembly           |
   +-------------------------------------------------------------+
```

Data flow per frame: `capture → inference → target selection → normalize → transmit`.

## 3. Project Structure

```
Tracker_1/
├── dataset/
│   ├── data.yaml                 # YOLO dataset config (auto-generated)
│   ├── images/{train,val}/       # training/validation images
│   └── labels/{train,val}/       # YOLO label files (normalized)
├── scripts/
│   ├── prepare_dataset.py        # raw -> train/val split + labels + data.yaml
│   └── train.py                  # YOLOv8 training entry point
├── src/
│   └── vision_module.py          # real-time detection & targeting module
├── models/
│   ├── yolov8n.pt                # pre-trained base weights (download once)
│   └── best.pt                   # trained weights (output of train.py)
 
├── runs/train/                   # training logs, curves, confusion matrix
├── logs/                         # vision module runtime logs
└── requirements.txt
```

## 4. Installation

### 4.1 Python environment (vision processor)

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux/macOS

# 2. CPU-only install
pip install -r requirements.txt

# 3. GPU install (CUDA 12.x) — install torch first, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# True = CUDA working; the vision module will use it automatically
```

### 4.2 Raspberry Pi side

The Pi only needs a serial/UDP listener. Example (Python 3 on the Pi):

```python
# pi_actuator.py — minimal receiver, reads JSON or CSV lines
import serial
ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)
while True:
    line = ser.readline().decode().strip()
    if not line:
        continue
    if line.startswith("{"):
        import json; x, y = json.loads(line)["x"], json.loads(line)["y"]
    else:
        x, y = map(float, line.split(","))
    # TODO: map (x, y) -> pan/tilt angles and drive servos
    print(f"Aim at x={x:.3f} y={y:.3f}")
```

## 5. Training Guide

### 5.1 Prepare raw data

Collect flea images (ideally 500+ with fleas visible, plus clean
backgrounds as negatives). Layout your raw data as:

```
raw_data/
├── images/            # *.jpg / *.png / *.webp
├── labels/            # optional — same filename as image, .txt or .xml
│                      #   YOLO:      "0 0.45 0.30 0.10 0.08"
│                      #   VOC XML:   auto-converted
│                      #   DarkLabel: "flea, x1,y1,x2,y2" CSV
└── classes.txt        # "flea" on line 0 (optional)
```

Then run:

```bash
python scripts/prepare_dataset.py --input raw_data --output dataset --split 0.8
```

This copies images into `dataset/images/{train,val}` (80/20, seeded random),
converts labels to YOLO format, generates `dataset/data.yaml` and validates
the result.

### 5.2 Get the pre-trained base weights

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('models/yolov8n.pt')"
```

### 5.3 Train

```bash
python scripts/train.py --epochs 150 --imgsz 640 --batch 16
```

Key flags: `--epochs`, `--imgsz` (640 default), `--batch`, `--device`
(`0` = GPU, `cpu`), `--augment` (stronger augmentation for small datasets),
`--resume` (continue an interrupted run).

**Outputs:** best weights → `models/best.pt`; plots, curves and confusion
matrix → `runs/train/flea_<timestamp>/`. Target metrics: `mAP50 > 0.8` for
a usable system.

## 6. Usage

### 6.1 USB camera

```bash
python src/vision_module.py --source 0 --model models/best.pt --port COM3
```

### 6.2 IP / RTSP camera

```bash
python src/vision_module.py --source "rtsp://user:pass@192.168.1.64:554/stream1" --host 192.168.1.50
```

### 6.3 Configuration

Flags or a JSON config file (`--config config.json`):

```json
{
  "source": 0,
  "model_path": "models/best.pt",
  "conf_thres": 0.35,
  "serial_port": "COM3",
  "baudrate": 115200,
  "send_json": true,
  "show_video": true
}
```

Press `q` in the video window to exit.

## 7. Communication Protocol

Vision → Raspberry Pi, one message per target update:

```
JSON:  {"x":0.45,"y":0.30}\n
CSV:   0.45,0.30\n
```

- Coordinates are **normalized to [0,1]**, relative to the full camera frame.
- A trailing `\n` terminates each message (delimiter).
- The vision module rate-limits sends (default 50 ms minimum interval) so
  the serial link and servos are never flooded.
- Serial default: `115200 baud, 8N1`. UDP fallback: port `5005`.

**Laser aiming math (Pi side):** with the laser mounted parallel to the
camera optical axis, pan/tilt can be mapped as
`angle_x = (x - 0.5) * FOV_x` and `angle_y = (0.5 - y) * FOV_y` (negate
y since image y grows downward), then feed into your servo controller.
Calibrate FOV per mounting once at install time.

## 8. Performance Tuning

| Goal               | Change                                                       |
|--------------------|--------------------------------------------------------------|
| Higher FPS         | `--imgsz 416`, enable GPU FP16 (auto), `--skip-frames 1`     |
| Better precision   | `--imgsz 640`+ , lower `--conf` to 0.25, train longer        |
| Less jitter        | Raise `min_detection_interval` (e.g. 0.1 s)                  |
| Lower camera delay | RTSP: reduce encoder GOP; USB: MJPG mode (set automatically) |
| Negative frames    | Ensure the dataset has ~20% background-only images           |

## 9. Troubleshooting

| Symptom                              | Fix                                                                 |
|--------------------------------------|---------------------------------------------------------------------|
| `CUDA not available` logged          | Install the CUDA torch build (see §4.1); verify `torch.cuda.is_available()`. |
| `Failed to open camera source: 0`    | Camera busy by another app; try `--source 1`; for IP cameras check the URL is reachable in a browser. |
| `Serial open failed (COM3)`          | Wrong port name (check Device Manager); port in use; add your user to the `dialout` group on Linux. |
| Model not found                       | Run training first (§5.3) or download base weights into `models/`.   |
| Servos never move                     | Confirm the Pi listener is running; test the port with a terminal at 115200 baud; check baudrate matches. |
| Low FPS / jitter                     | See §8; verify GPU is used (`python -c "import torch; print(torch.cuda.is_available())"`). |
| Missing label files                  | Labels must share the image's filename in `raw_data/labels/`; run `prepare_dataset.py` again. |
| Dataset validation FAILED            | Read the printed problems — usually missing labels or an empty split. |

## 10. Safety

- The laser assembly must use an eye-safe power class and interlock.
- Never point the laser at people or animals' eyes.
- Add a mechanical kill-switch to the Pi's GPIO as a hardware safety stop.

---

*Built with YOLOv8 (Ultralytics), OpenCV and PySerial.*