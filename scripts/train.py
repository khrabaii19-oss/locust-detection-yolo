#!/usr/bin/env python3
"""
train.py
========
Trains a YOLOv8 model for flea detection.

  - Loads a pre-trained YOLOv8n checkpoint from models/ (e.g. models/yolov8n.pt)
  - Trains on the dataset described by dataset/data.yaml
  - Saves the best weights to models/best.pt
  - Runs on GPU (CUDA) automatically when available
  - Exports training metrics and plots into runs/ and models/

Usage:
    python scripts/train.py --epochs 150 --imgsz 640 --batch 16

Pre-requisites:
    - pip install -r requirements.txt
    - dataset prepared with scripts/prepare_dataset.py
    - pre-trained checkpoint at models/yolov8n.pt
      (download with: yolo download or:
       python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')")
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with a consistent format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 for flea detection.")
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Path to dataset YAML (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--weights",
        default="models/yolov8n.pt",
        help="Pre-trained weights to fine-tune from (default: models/yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: '0' for CUDA GPU 0, 'cpu', or '0,1' for multi-GPU (default: auto)",
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        help="Directory for training runs (default: runs/train)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name (default: auto-generated 'flea_<timestamp>')",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience in epochs (default: 30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data-loader workers (default: 8)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint of the same run",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable stronger augmentation (good for small datasets)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Path:
    """
    Run the YOLOv8 training loop and return the path to the best weights.

    Raises SystemExit(1) if the dataset or pre-trained weights are missing.
    """
    logging.info("=== YOLOv8 Flea Detection Training ===")

    # --- Validate inputs before spending time on model load ------------------
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logging.error(f"Dataset YAML not found: {data_yaml}")
        logging.error("Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logging.error(f"Pre-trained weights not found: {weights_path}")
        logging.error(
            "Download yolov8n.pt into models/ with:\n"
            f"  python -c \"from ultralytics import YOLO; "
            f"YOLO('yolov8n.pt').save('{weights_path}')\""
        )
        sys.exit(1)

    # --- Import ultralytics lazily so --help works without the package ------
    try:
        from ultralytics import YOLO
    except ImportError:
        logging.error(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        )
        sys.exit(1)

    # --- Load the pre-trained model ------------------------------------------
    logging.info(f"Loading pre-trained model: {weights_path}")
    model = YOLO(str(weights_path))

    # --- Run name with timestamp ----------------------------------------------
    import datetime
    run_name = args.name or f"flea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # --- Training arguments ----------------------------------------------------
    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": run_name,
        "patience": args.patience,
        "workers": args.workers,
        "resume": args.resume,
        "plots": True,          # save loss/PR curves
        "save_period": -1,      # only save best + last
        "val": True,
    }
    if args.augment:
        train_args.update({"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                           "degrees": 0.0, "translate": 0.1, "scale": 0.5,
                           "fliplr": 0.5, "mosaic": 1.0})

    logging.info(
        f"Starting training: epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, device={args.device or 'auto'}"
    )

    # --- Train ------------------------------------------------------------------
    results = model.train(**train_args)

    # --- Locate the best weights produced by the run -----------------------------
    run_dir = Path(args.project) / run_name
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.exists():
        # Older/newer ultralytics versions may nest the run under
        # 'runs/detect/...' - search the whole runs tree as a fallback.
        matches = sorted(Path("runs").rglob(f"**/{run_name}/weights/best.pt"))
        if matches:
            best_weights = matches[-1]
        else:
            logging.error("Training finished but best.pt was not found.")
            sys.exit(1)

    # --- Copy best weights to the canonical location models/best.pt -------------
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    destination = models_dir / "best.pt"
    import shutil
    shutil.copy2(best_weights, destination)
    logging.info(f"Best weights saved to: {destination}")

    # --- Summary -----------------------------------------------------------------
    logging.info("=== Training complete ===")
    logging.info(f"Results (curves, confusion matrix): {run_dir}")
    if hasattr(results, "metrics"):
        m = results.metrics
        logging.info(
            f"mAP50: {getattr(m, 'map50', float('nan')):.4f} | "
            f"mAP50-95: {getattr(m, 'map', float('nan')):.4f}"
        )
    return destination


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    try:
        train(args)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - report any training failure
        logging.error(f"Training failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
