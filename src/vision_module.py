#!/usr/bin/env python3
"""
vision_module.py
================
Real-time flea detection + targeting module for the YOLOv8 laser system.

Pipeline:
    1. Load the trained model (models/best.pt)
    2. Capture frames from a USB camera or RTSP/HTTP IP camera
    3. Run YOLOv8 inference at reduced resolution for low latency
    4. Pick the closest target (largest bounding box = nearest flea)
    5. Normalize the target center to (0-1) coordinates
    6. Send the coordinates to the Raspberry Pi actuator via serial
       (or optional network socket) using the JSON or CSV protocol:
           {"x":0.45,"y":0.30}   or   "0.45,0.30"
    7. Display annotated frames with FPS overlay (optional)

Threading model:
    - Capture thread reads frames from the camera as fast as possible.
    - Main loop consumes the latest frame, runs inference and transmits.
    This decouples camera latency from inference latency.

Usage:
    python src/vision_module.py --source 0 --model models/best.pt --port COM3

Configuration can also be supplied via environment variables or a
config.json (see --config).
"""

import argparse
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional imports with graceful degradation
# ---------------------------------------------------------------------------
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import serial  # type: ignore
except ImportError:
    serial = None

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Runtime configuration for the vision module."""

    source: str | int = 0                    # camera index or RTSP/HTTP URL
    model_path: str = "models/best.pt"       # trained YOLO weights
    conf_thres: float = 0.35                 # detection confidence threshold
    iou_thres: float = 0.45                  # NMS IoU threshold
    imgsz: int = 640                         # inference image size
    # --- serial (Raspberry Pi actuator) ---
    serial_port: str | None = None           # e.g. "COM3" (Win) / "/dev/ttyUSB0" (Linux)
    baudrate: int = 115200
    # --- optional network fallback ---
    network_host: str | None = None          # e.g. "192.168.1.50"
    network_port: int = 5005
    network_enabled: bool = False
    # --- behavior ---
    send_json: bool = True                   # True -> {"x":..,"y":..}, False -> "x,y"
    min_detection_interval: float = 0.05     # seconds between transmissions
    send_timeout: float = 0.05               # serial write timeout (s)
    show_video: bool = True                  # display annotated frames
    skip_frames: int = 0                     # process every (N+1)-th frame
    log_file: str | None = "logs/vision.log"

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build a Config from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(log_file: str | None, verbose: bool = False) -> None:
    """Configure logging to console and optional file."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------
class CameraStream:
    """
    Background thread that continuously grabs frames from a camera.

    Supports:
        - USB webcam:          source = 0, 1, ...
        - RTSP:                "rtsp://user:pass@ip:554/stream"
        - HTTP video:          "http://ip:8080/video"

    The latest frame is stored in a shared slot; inference code calls
    read() to fetch it. If a new frame has not arrived yet, read()
    returns the previous one (never blocks inference).
    """

    def __init__(self, source: str | int, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self._cap = None
        self._frame: object | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._fps = 0.0
        self._n_frames = 0
        self._last_time = time.perf_counter()
        self._connected = False

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        """Open the camera and launch the capture thread."""
        if cv2 is None:
            raise RuntimeError("OpenCV is required. Run: pip install -r requirements.txt")

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")

        # Best-effort resolution / latency tuning (ignored when unsupported).
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Lower buffering on RTSP streams to cut latency.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._connected = True
        logging.info(f"Camera opened: {self.source}")

    def _run(self) -> None:
        """Capture loop - runs until stop() is called."""
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # Transient failures are common with network cameras.
                logging.warning("Camera read failed, retrying...")
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._n_frames += 1
            # FPS of the *capture* side, updated once per second.
            now = time.perf_counter()
            if now - self._last_time >= 1.0:
                self._fps = self._n_frames / (now - self._last_time)
                self._n_frames = 0
                self._last_time = now

    def read(self):
        """Return the latest frame (or None if none captured yet)."""
        with self._lock:
            return self._frame

    def fps(self) -> float:
        """Capture-side FPS."""
        return self._fps

    def connected(self) -> bool:
        """True when the capture thread is running."""
        return self._connected

    def stop(self) -> None:
        """Stop capture and release the camera."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self._connected = False
        logging.info("Camera released")


# ---------------------------------------------------------------------------
# Actuator communication (serial + network)
# ---------------------------------------------------------------------------
class ActuatorLink:
    """
    Sends normalized (0-1) target coordinates to the Raspberry Pi.

    Protocols (selected via Config.send_json):
        JSON:  b'{"x":0.45,"y":0.30}\n'
        CSV:   b'0.45,0.30\n'

    Backends:
        - Serial port (USB-UART bridge to the Pi or directly to the servo driver)
        - UDP/TCP network socket (fallback, e.g. WiFi link to the Pi)

    A trailing newline acts as the frame delimiter for the receiver.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ser = None
        self._sock = None
        self._last_send = 0.0
        self._msg_count = 0
        self._send_errors = 0

    # -- setup ----------------------------------------------------------------
    def connect(self) -> None:
        """Open the selected transport(s). Raises if none can be opened."""
        connected = False

        if self.cfg.serial_port and serial is not None:
            try:
                self._ser = serial.Serial(
                    port=self.cfg.serial_port,
                    baudrate=self.cfg.baudrate,
                    timeout=self.cfg.send_timeout,
                    write_timeout=self.cfg.send_timeout,
                )
                connected = True
                logging.info(f"Serial connected: {self.cfg.serial_port} @ {self.cfg.baudrate}")
            except (OSError, ValueError) as exc:
                logging.error(f"Serial open failed ({self.cfg.serial_port}): {exc}")

        if self.cfg.network_enabled and self.cfg.network_host:
            try:
                self._sock = self._make_socket()
                connected = True
                logging.info(f"Network socket ready: {self.cfg.network_host}:{self.cfg.network_port}")
            except OSError as exc:
                logging.error(f"Socket setup failed: {exc}")

        if not connected:
            logging.warning(
                "No actuator transport configured - running in 'detection only' mode."
            )

    def _make_socket(self):
        """Create a UDP socket (low latency, no handshake)."""
        sock = __import__("socket").socket(
            __import__("socket").AF_INET, __import__("socket").SOCK_DGRAM
        )
        sock.settimeout(0.01)
        return sock

    # -- encoding --------------------------------------------------------------
    def encode(self, x: float, y: float) -> bytes:
        """Encode normalized coordinates according to the chosen protocol."""
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        if self.cfg.send_json:
            payload = json.dumps({"x": round(x, 4), "y": round(y, 4)})
        else:
            payload = f"{x:.4f},{y:.4f}"
        return (payload + "\n").encode("ascii")

    # -- send ------------------------------------------------------------------
    def send(self, x: float, y: float) -> bool:
        """
        Transmit target coordinates, honoring the minimum send interval.
        Returns True when a message was actually transmitted.
        """
        now = time.perf_counter()
        if now - self._last_send < self.cfg.min_detection_interval:
            return False
        self._last_send = now

        data = self.encode(x, y)
        ok = False

        if self._ser is not None:
            try:
                self._ser.write(data)
                ok = True
            except OSError as exc:
                self._send_errors += 1
                logging.error(f"Serial write failed: {exc}")

        if self._sock is not None and self.cfg.network_host:
            try:
                self._sock.sendto(data, (self.cfg.network_host, self.cfg.network_port))
                ok = True
            except OSError as exc:
                self._send_errors += 1
                logging.error(f"Socket send failed: {exc}")

        if ok:
            self._msg_count += 1
            logging.debug(f"Sent: {data.rstrip()}")
        return ok

    # -- stats / cleanup --------------------------------------------------------
    def stats(self) -> dict:
        """Return transmission statistics for the FPS overlay."""
        return {"sent": self._msg_count, "errors": self._send_errors}

    def close(self) -> None:
        """Release serial port and socket."""
        if self._ser:
            try:
                self._ser.close()
            except OSError:
                pass
        if self._sock:
            self._sock.close()
        logging.info("Actuator link closed")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def bbox_center(box) -> tuple[float, float]:
    """
    Compute the center of a YOLO result box (xywh format).
    Returns (cx, cy) in *pixel* coordinates.
    """
    x, y, w, h = box.xywh[0].cpu().numpy()
    return float(x), float(y)


def bbox_area(box) -> float:
    """Area of a detection box in pixels - used to pick the nearest flea."""
    _, _, w, h = box.xywh[0].cpu().numpy()
    return float(w * h)


def normalize(cx: float, cy: float, frame_w: int, frame_h: int) -> tuple[float, float]:
    """
    Convert pixel center coordinates to normalized 0-1 coordinates.

    YOLO convention: origin is the top-left corner of the image,
    so (0.5, 0.5) is the frame center - ideal for aiming a laser
    mounted at the camera optical axis.
    """
    return cx / frame_w, cy / frame_h


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------
class VisionModule:
    """Orchestrates camera capture, inference, targeting and transmission."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.cam = None
        self.link = None
        self._device = "cpu"
        self._running = False
        self._fps = 0.0
        self._frame_count = 0
        self._detection_count = 0
        self._targets: list[dict] = []   # latest frame's detections (json-serializable)

    # -- setup ----------------------------------------------------------------
    def load_model(self) -> None:
        """Load the trained YOLO model and select the compute device."""
        if not Path(self.cfg.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.cfg.model_path}. "
                "Run scripts/train.py first."
            )
        from ultralytics import YOLO  # lazy import (slow)

        self.model = YOLO(self.cfg.model_path)
        self._device = self._pick_device()
        logging.info(f"Model loaded: {self.cfg.model_path} (device: {self._device})")

    def _pick_device(self) -> str:
        """Prefer CUDA GPU, fall back to CPU."""
        import torch
        if torch.cuda.is_available():
            return "0"
        logging.warning("CUDA not available - using CPU (expect lower FPS)")
        return "cpu"

    # -- main loop --------------------------------------------------------------
    def run(self) -> None:
        """Start camera + actuator, then run the real-time loop."""
        if cv2 is None:
            raise RuntimeError("OpenCV is required. Run: pip install -r requirements.txt")

        self.load_model()

        # Configure the camera and inference size.
        source = self.cfg.source
        try:
            source = int(source)  # camera index
        except (TypeError, ValueError):
            pass  # keep as URL string

        self.cam = CameraStream(source)
        self.cam.start()

        self.link = ActuatorLink(self.cfg)
        self.link.connect()

        self._running = True
        logging.info("Vision module started. Press 'q' in the video window to quit.")

        # Pre-allocate the input size for the model once.
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            self.shutdown()

    def _run_loop(self) -> None:
        """The real-time inference loop."""
        from ultralytics.utils.plotting import Annotator  # lazy import

        window_name = "Flea Detection"
        frame_idx = 0
        prev_time = time.perf_counter()
        fps_ema = 0.0

        while self._running:
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Optional frame skipping for very fast cameras.
            if self.cfg.skip_frames > 0 and frame_idx % (self.cfg.skip_frames + 1) != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            h, w = frame.shape[:2]

            # --- inference ----------------------------------------------------
            results = self.model.predict(
                frame,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                imgsz=self.cfg.imgsz,
                device=self._device,
                verbose=False,      # keep the console quiet
            )
            boxes = results[0].boxes
            self._frame_count += 1

            # --- target selection ----------------------------------------------
            target, self._targets = self._select_target(boxes, w, h)

            # --- transmit -------------------------------------------------------
            if target is not None:
                self._detection_count += 1
                self.link.send(target["x"], target["y"])

            # --- optional visualization ------------------------------------------
            if self.cfg.show_video:
                frame = self._annotate(frame, boxes, target, fps_ema)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False

            # --- FPS (smoothed) ---------------------------------------------------
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps
                self._fps = fps_ema

            if frame_idx % 30 == 0:
                logging.info(
                    f"fps={self._fps:.1f} detections={len(boxes)} "
                    f"target={target and 'YES' or 'none'}"
                )

    # -- targeting ---------------------------------------------------------------
    def _select_target(self, boxes, frame_w: int, frame_h: int):
        """
        Choose the best flea to shoot.

        Strategy: the closest object to the laser is usually the one with the
        largest bounding-box area (perspective rule of thumb for similar
        objects). All fleas are kept in `targets`; only the largest is sent.

        Returns (target_dict, all_targets_list).
        Each target dict: {"x": 0-1, "y": 0-1, "conf": ..., "area": ...}
        """
        all_targets = []
        best = None
        best_area = -1.0

        for box in boxes:
            cx, cy = bbox_center(box)
            area = bbox_area(box)
            nx, ny = normalize(cx, cy, frame_w, frame_h)
            conf = float(box.conf[0].cpu().numpy())
            entry = {"x": nx, "y": ny, "conf": round(conf, 4), "area": round(area, 1)}
            all_targets.append(entry)
            if area > best_area:
                best_area = area
                best = entry

        return best, all_targets

    def _annotate(self, frame, boxes, target, fps: float) -> object:
        """Draw detection boxes, target crosshair and FPS on the frame."""
        from ultralytics.utils.plotting import Annotator

        annotator = Annotator(frame, line_width=2)
        for box in boxes:
            annotator.box_label(box.xyxy[0].cpu().numpy(), "flea", color=(0, 255, 0))

        if target is not None:
            h, w = frame.shape[:2]
            px, py = int(target["x"] * w), int(target["y"] * h)
            cv2.circle(frame, (px, py), 6, (0, 0, 255), 2)
            cv2.line(frame, (px - 12, py), (px + 12, py), (0, 0, 255), 1)
            cv2.line(frame, (px, py - 12), (px, py + 12), (0, 0, 255), 1)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return frame

    # -- lifecycle ----------------------------------------------------------------
    def shutdown(self) -> None:
        """Release camera, actuator link and windows."""
        self._running = False
        if self.cam:
            self.cam.stop()
        if self.link:
            self.link.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
        logging.info(
            f"Session stats: frames={self._frame_count}, detections={self._detection_count}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time flea detection & targeting")
    parser.add_argument("--source", default=0, help="Camera index (0,1..) or RTSP/HTTP URL")
    parser.add_argument("--model", dest="model_path", default="models/best.pt", help="Trained model path")
    parser.add_argument("--port", dest="serial_port", default=None, help="Serial port (e.g. COM3)")
    parser.add_argument("--baud", dest="baudrate", type=int, default=115200)
    parser.add_argument("--conf", dest="conf_thres", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--csv", dest="send_json", action="store_false", help="Send CSV instead of JSON")
    parser.add_argument("--host", dest="network_host", default=None, help="UDP target host (fallback to serial)")
    parser.add_argument("--no-video", dest="show_video", action="store_false", help="Disable display")
    parser.add_argument("--config", default=None, help="JSON config file (overrides defaults)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Entry point."""
    args = parse_args(argv)

    # Merge: defaults <- config file <- CLI flags.
    cfg_data = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            logging.error(f"Config file not found: {cfg_path}")
            return
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))

    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k not in ("config", "verbose")}
    cfg = Config.from_dict({**cfg_data, **cli_overrides})

    setup_logging(cfg.log_file, args.verbose)
    try:
        module = VisionModule(cfg)
        module.run()
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        logging.error(f"Startup failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
