#!/usr/bin/env python3
"""
fix_labels_complete.py
======================
Fixes broken YOLO label files in the dataset:
  - Rewrites every class id to 0 (single-class "flea" dataset)
  - Detects oversized bounding boxes (covering a large fraction of the
    image) and shrinks them to ~15-20% of the image, keeping their center.

Usage:
    python scripts/fix_labels_complete.py
    python scripts/fix_labels_complete.py --labels dataset/labels

Requires only the standard library.
"""

import argparse
import logging
from pathlib import Path

DEFAULT_LABELS_DIR = Path("dataset/labels")

# A box is "huge" when its area covers more than 25% of the image or
# either side spans more than half the image.
AREA_THRESHOLD = 0.25
SIDE_THRESHOLD = 0.5

# Target size for oversized boxes (15-20% of the image).
TARGET_SIZE = 0.18


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def is_huge(w: float, h: float) -> bool:
    return w * h > AREA_THRESHOLD or w > SIDE_THRESHOLD or h > SIDE_THRESHOLD


def fix_label_file(label_path: Path) -> tuple[int, int, int]:
    """
    Rewrite one YOLO label file in place.
    Returns (lines_fixed, boxes_resized, files_changed).
    """
    original_lines = label_path.read_text(encoding="utf-8").splitlines()
    if not original_lines:
        return 0, 0, 0

    fixed = 0
    resized = 0
    out_lines = []
    for raw in original_lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            logging.warning(f"{label_path}: skipping malformed line: {raw}")
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])

        if cls != 0:
            cls = 0
            fixed += 1

        if is_huge(w, h):
            w = TARGET_SIZE
            h = TARGET_SIZE
            resized += 1

        out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    new_text = "\n".join(out_lines) + ("\n" if out_lines else "")
    if new_text != label_path.read_text(encoding="utf-8"):
        label_path.write_text(new_text, encoding="utf-8")
        return fixed, resized, 1
    return fixed, resized, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix YOLO labels: force class 0 and shrink huge boxes."
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_DIR),
        help="Directory containing YOLO label .txt files (default: dataset/labels)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    labels_dir = Path(args.labels)
    if not labels_dir.is_dir():
        logging.error(f"Labels directory not found: {labels_dir}")
        raise SystemExit(1)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        logging.warning(f"No .txt label files found in {labels_dir}")
        return

    total_fixed = total_resized = files_changed = 0
    for label_path in label_files:
        fixed, resized, changed = fix_label_file(label_path)
        total_fixed += fixed
        total_resized += resized
        files_changed += changed
        if fixed or resized:
            logging.info(
                f"{label_path.name}: classes fixed={fixed}, boxes resized={resized}"
            )

    logging.info(
        f"Done: {files_changed}/{len(label_files)} files changed, "
        f"{total_fixed} class id(s) set to 0, {total_resized} huge box(es) resized"
    )


if __name__ == "__main__":
    main()