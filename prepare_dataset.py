#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Organizes raw flea images into a YOLO-ready dataset:
  - Splits images into train/val (80/20) with reproducible seed
  - Converts any bounding-box labels found alongside the raw images
    into YOLO format (normalized x_center, y_center, width, height)
  - Generates dataset/data.yaml automatically
  - Validates the final dataset structure

Expected raw input layout:
    raw_data/
        images/          # jpg/png/jpeg images
        labels/          # optional: one .txt per image with labels.
                         #   Accepted formats:
                         #     a) YOLO:  "class x_center y_center w h" (already normalized)
                         #     b) PASCAL VOC XML / DarkLabel CSV: see --label-format
        classes.txt      # optional: class names, one per line (flea => line 0)

If no labels are provided, the script creates empty label files so the
dataset structure is still valid for training.

Usage:
    python scripts/prepare_dataset.py \
        --input raw_data \
        --output dataset \
        --split 0.8 \
        --seed 42 \
        --label-format yolo

Requires only the standard library (plus PyYAML for yaml output, optional).
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_LABEL_EXTENSIONS = {".txt", ".xml"}

# Supported raw label formats
FORMAT_YOLO = "yolo"      # already-normalized txt: class cx cy w h
FORMAT_VOC = "voc"        # PASCAL VOC XML
FORMAT_DARKLABEL = "darklabel"  # DarkLabel CSV: class x1 y1 x2 y2

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger with a consistent format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def iter_images(image_dir: Path, exclude: set[str] | None = None) -> list[Path]:
    """Return a sorted list of image files inside a directory.

    `exclude` holds subdirectory names that must be skipped (e.g. the
    'train'/'val' output dirs when input and output are the same folder).
    """
    exclude = exclude or set()
    return sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        and not any(part in exclude for part in p.relative_to(image_dir).parts)
    )


def find_label_file(image_path: Path, labels_dir: Path) -> Path | None:
    """Match an image to its label file (same stem, .txt or .xml)."""
    for ext in VALID_LABEL_EXTENSIONS:
        candidate = labels_dir / f"{image_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_yolo_label(label_path: Path, num_classes: int) -> list[str]:
    """
    Parse an already-normalized YOLO label file.
    Returns a list of normalized YOLO lines (one per object).
    Raises ValueError on malformed or out-of-range values.
    """
    lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{line_no} expected 5 values, got {len(parts)}"
                )
            cls, cx, cy, w, h = (int(parts[0]), *map(float, parts[1:]))
            if not (0 <= cls < num_classes):
                raise ValueError(f"{label_path}:{line_no} class {cls} out of range")
            for name, value in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                if not (0.0 <= value <= 1.0):
                    raise ValueError(
                        f"{label_path}:{line_no} {name}={value} not in [0,1]"
                    )
            lines.append(line)
    return lines


def parse_voc_label(label_path: Path, num_classes: int, class_names: dict) -> list[str]:
    """
    Parse a PASCAL VOC XML annotation and convert to YOLO format.
    Requires cv2 to read image dimensions; falls back to YOLO-style
    normalized values if w/h are already present as attributes.
    """
    import re

    try:
        import cv2  # type: ignore
    except ImportError:
        raise RuntimeError("cv2 is required for --label-format voc")

    xml_text = label_path.read_text(encoding="utf-8", errors="replace")
    image_match = re.search(r"<filename>(.*?)</filename>", xml_text, re.S)

    # Fallback: if "width"/"height" elements exist, use them.
    w_match = re.search(r"<width>(\d+)</width>", xml_text)
    h_match = re.search(r"<height>(\d+)</height>", xml_text)

    image_path = None
    if image_match:
        image_path = label_path.parent / image_match.group(1).strip()

    if (w_match and h_match) and image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = int(w_match.group(1)), int(h_match.group(1))
    else:
        raise ValueError(
            f"{label_path}: cannot determine image size for VOC conversion"
        )

    objects = re.findall(
        r"<object>.*?<name>(.*?)</name>.*?"
        r"<bndbox>.*?<xmin>(\d+)</xmin>.*?<ymin>(\d+)</ymin>.*?"
        r"<xmax>(\d+)</xmax>.*?<ymax>(\d+)</ymax>.*?</bndbox>",
        xml_text,
        re.S,
    )

    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        cls = class_names.get(name.strip())
        if cls is None:
            logging.warning(f"{label_path}: unknown class '{name}', skipping")
            continue
        x1, y1, x2, y2 = map(float, (xmin, ymin, xmax, ymax))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, y1, x2, y2 = (
            max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2),
        )
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def parse_darklabel_csv(label_path: Path, num_classes: int, class_names: dict) -> list[str]:
    """
    Parse a DarkLabel CSV annotation:
        class, x1, y1, x2, y2 [, ...]
    Image size is read with cv2; the matching image must be present
    next to the label file or in the images directory.
    """
    import csv

    try:
        import cv2  # type: ignore
    except ImportError:
        raise RuntimeError("cv2 is required for --label-format darklabel")

    # Try to find the corresponding image to read dimensions.
    image_candidates = [
        label_path.with_suffix(".jpg"),
        label_path.with_suffix(".jpeg"),
        label_path.with_suffix(".png"),
    ]
    img_path = next((p for p in image_candidates if p.exists()), None)
    if img_path is None:
        raise ValueError(f"{label_path}: no matching image found for DarkLabel CSV")
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"{label_path}: failed to read image {img_path}")
    img_h, img_w = img.shape[:2]

    lines = []
    with open(label_path, "r", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            name, x1, y1, x2, y2 = row[0], *map(float, row[1:5])
            cls = class_names.get(name.strip())
            if cls is None:
                logging.warning(f"{label_path}: unknown class '{name}', skipping")
                continue
            x1, x2 = sorted((max(0, x1), min(img_w, x2)))
            y1, y2 = sorted((max(0, y1), min(img_h, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def load_class_names(classes_file: Path | None, fallback: str = "flea") -> dict[str, int]:
    """
    Build a name -> id mapping. When no classes.txt exists, returns a
    single-class mapping {fallback: 0}.
    """
    if classes_file and classes_file.exists():
        names = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if names:
            return {name: idx for idx, name in enumerate(names)}
    return {fallback: 0}


def convert_labels(
    image_path: Path,
    label_path: Path | None,
    num_classes: int,
    class_names: dict,
    label_format: str,
) -> list[str]:
    """Dispatch label conversion based on the chosen format."""
    if label_path is None:
        return []  # no labels -> empty label file (image used as background)

    if label_format == FORMAT_YOLO:
        return parse_yolo_label(label_path, num_classes)
    if label_format == FORMAT_VOC:
        return parse_voc_label(label_path, num_classes, class_names)
    if label_format == FORMAT_DARKLABEL:
        return parse_darklabel_csv(label_path, num_classes, class_names)
    raise ValueError(f"Unsupported label format: {label_format}")


def write_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    """Generate dataset/data.yaml pointing at the train/val image dirs."""
    names = {i: name for i, name in enumerate(class_names)}
    yaml_content = (
        "# Auto-generated by scripts/prepare_dataset.py - do not edit manually\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        + "\n".join(f"  {k}: {v}" for k, v in names.items())
        + "\n"
    )
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml_content, encoding="utf-8")
    logging.info(f"Generated {data_yaml}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def prepare_dataset(args) -> None:
    """Orchestrate the whole dataset preparation pipeline."""
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    train_ratio = float(args.split)
    num_classes = int(args.num_classes)

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--split must be between 0 and 1")

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    classes_file = input_dir / "classes.txt"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # --- Class names -------------------------------------------------------
    class_names = load_class_names(classes_file, args.class_name)
    # If a single-class dataset is assumed, force the requested class name.
    if args.class_name and args.class_name not in class_names:
        class_names = {args.class_name: 0}
    logging.info(f"Classes: {class_names}")

    # --- Collect images ----------------------------------------------------
    output_subdirs = {"train", "val"} if output_dir == input_dir else set()
    images = iter_images(images_dir, exclude=output_subdirs)
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}")
    logging.info(f"Found {len(images)} images")

    # --- Deterministic 80/20 split -----------------------------------------
    random.seed(args.seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]

    # --- Build output structure ---------------------------------------------
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"
    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Copy images + convert labels ---------------------------------------
    copied = 0
    errors = []
    for subset, img_list in (("train", train_images), ("val", val_images)):
        for image_path in img_list:
            label_path = find_label_file(image_path, labels_dir)
            try:
                yolo_lines = convert_labels(
                    image_path,
                    label_path,
                    num_classes,
                    class_names,
                    args.label_format,
                )
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))
                logging.error(str(exc))
                continue

            dst_img = (train_img_dir if subset == "train" else val_img_dir) / image_path.name
            dst_lbl = (train_lbl_dir if subset == "train" else val_lbl_dir) / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dst_img)
            dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            copied += 1
        logging.info(f"{subset}: {len(img_list)} images processed")

    if errors:
        logging.warning(f"{len(errors)} files skipped due to errors")

    # --- Generate data.yaml --------------------------------------------------
    write_data_yaml(output_dir, sorted(class_names, key=class_names.get))

    # --- Validate -------------------------------------------------------------
    validate_dataset(output_dir)
    logging.info(f"Dataset ready at {output_dir.resolve()}")


def validate_dataset(output_dir: Path) -> None:
    """
    Verify the generated dataset is structurally sound:
      - every image in images/* has a sibling label in labels/*
      - every label references a class id < nc
      - train/val directories are non-empty
    Prints a short summary. Raises SystemExit(1) on critical failures.
    """
    problems = []
    for subset in ("train", "val"):
        img_dir = output_dir / "images" / subset
        lbl_dir = output_dir / "labels" / subset
        if not img_dir.exists() or not any(img_dir.iterdir()):
            problems.append(f"no images in {img_dir}")
            continue

        imgs = iter_images(img_dir)
        for img in imgs:
            label_file = lbl_dir / f"{img.stem}.txt"
            if not label_file.exists():
                problems.append(f"missing label for {img.name}")
                continue
            for line_no, raw in enumerate(
                label_file.read_text(encoding="utf-8").splitlines(), 1
            ):
                parts = raw.split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    if cls < 0:
                        problems.append(f"{label_file}:{line_no} negative class id")

    train_count = len(iter_images(output_dir / "images" / "train"))
    val_count = len(iter_images(output_dir / "images" / "val"))
    logging.info(f"Validation summary -> train: {train_count}, val: {val_count}")
    if problems:
        for p in problems[:20]:
            logging.warning(f"Validation: {p}")
        logging.warning(f"{len(problems)} validation problem(s) found")
    if not problems and train_count > 0 and val_count > 0:
        logging.info("Dataset validation PASSED")
    else:
        logging.error("Dataset validation FAILED")
        sys.exit(1)


def main() -> None:
    """Entry point: parse CLI args and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare a YOLOv8 flea-detection dataset from raw images."
    )
    parser.add_argument("--input", default="dataset", help="Raw data directory (images/, labels/, classes.txt)")
    parser.add_argument("--output", default="dataset", help="Output dataset directory")
    parser.add_argument("--split", default=0.8, type=float, help="Train fraction (default 0.8)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--label-format",
        choices=[FORMAT_YOLO, FORMAT_VOC, FORMAT_DARKLABEL],
        default=FORMAT_YOLO,
        help="Format of raw annotations (default: yolo)",
    )
    parser.add_argument("--class-name", default="flea", help="Class name when classes.txt is absent")
    parser.add_argument("--num-classes", default=1, type=int, help="Number of classes")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    try:
        prepare_dataset(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error(f"Preparation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
