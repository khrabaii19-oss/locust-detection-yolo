#!/usr/bin/env python3
"""
run_all.py
==========
Runs the complete pipeline in one shot:
    1. scripts/fix_labels_complete.py      (classes -> 0, shrink huge boxes)
    2. scripts/prepare_dataset.py          (train/val split + data.yaml)
    3. scripts/train.py --epochs 50        (train, save models/best.pt)
    4. src/vision_module.py                (live detection / targeting)

Usage:
    python scripts/run_all.py [--epochs 50] [--no-vision]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = sys.executable


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run_step(name: str, script: str, *args: str) -> None:
    logging.info(f"=== Step {name}: {script} ===")
    cmd = [str(PYTHON), str(Path(script)), *args]
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        logging.error(f"Step {name} FAILED (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)
    logging.info(f"Step {name} completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full flea pipeline.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip the live vision module (camera) at the end",
    )
    args = parser.parse_args()

    setup_logging()
    steps = [
        ("1/4", "scripts/fix_labels_complete.py"),
        ("2/4", "scripts/prepare_dataset.py"),
        ("3/4", "scripts/train.py", "--epochs", str(args.epochs)),
    ]
    if not args.no_vision:
        steps.append(("4/4", "src/vision_module.py"))

    for step in steps:
        run_step(step[0], step[1], *step[2:])

    logging.info("=== Full pipeline finished ===")


if __name__ == "__main__":
    main()