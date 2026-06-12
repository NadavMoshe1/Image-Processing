"""BDD100K label loading and sample selection utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.paths import IMAGES_ROOT, LABELS_ROOT, SEG_COLOR_ROOT, SEG_ID_ROOT

# Cityscapes / BDD100K shared trainId names (0–18); 255 = ignore
SEG_CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# Map YOLO COCO class names → BDD100K detection JSON categories
YOLO_TO_BDD_CATEGORY = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motor",
    "bicycle": "bike",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
}

BDD_DETECTION_COLORS = {
    "car": (0, 255, 0),
    "truck": (255, 128, 0),
    "bus": (255, 128, 0),
    "person": (0, 128, 255),
    "rider": (255, 0, 255),
    "bike": (255, 0, 255),
    "motor": (255, 0, 255),
    "traffic light": (0, 255, 255),
    "traffic sign": (255, 255, 0),
    "train": (128, 0, 255),
}


def build_label_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for label_path in (LABELS_ROOT / split).glob("*.json"):
            index[label_path.stem] = label_path
    return index


def resolve_label_path(stem: str, label_index: dict[str, Path]) -> Path | None:
    return label_index.get(stem)


def load_detection_boxes(label_path: Path | None) -> list[dict]:
    if label_path is None or not label_path.exists():
        return []
    with label_path.open(encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            if "box2d" not in obj:
                continue
            box = obj["box2d"]
            boxes.append(
                {
                    "category": obj.get("category", "unknown"),
                    "x1": float(box["x1"]),
                    "y1": float(box["y1"]),
                    "x2": float(box["x2"]),
                    "y2": float(box["y2"]),
                }
            )
    return boxes


def seg_id_path(split: str, stem: str) -> Path:
    return SEG_ID_ROOT / split / f"{stem}_train_id.png"


def seg_color_path(split: str, stem: str) -> Path:
    return SEG_COLOR_ROOT / split / f"{stem}_train_color.png"


def load_seg_mask(split: str, stem: str) -> np.ndarray | None:
    path = seg_id_path(split, stem)
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return mask


def load_seg_color(split: str, stem: str) -> np.ndarray | None:
    path = seg_color_path(split, stem)
    if not path.exists():
        return None
    return cv2.imread(str(path))


def has_detection_and_seg(split: str, stem: str, label_index: dict[str, Path]) -> bool:
    label_path = label_index.get(stem)
    if label_path is None or not load_detection_boxes(label_path):
        return False
    return seg_id_path(split, stem).exists()


def select_paired_images(
    split: str,
    num_images: int,
    seed: int,
    label_index: dict[str, Path] | None = None,
) -> list[Path]:
    label_index = label_index or build_label_index()
    candidates = [
        p
        for p in sorted((IMAGES_ROOT / split).glob("*.jpg"))
        if has_detection_and_seg(split, p.stem, label_index)
    ]
    if not candidates:
        return []
    n = min(num_images, len(candidates))
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, n))


def draw_detection_boxes(image: np.ndarray, boxes: list[dict], prefix: str = "") -> np.ndarray:
    vis = image.copy()
    for box in boxes:
        color = BDD_DETECTION_COLORS.get(box["category"], (255, 255, 255))
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix}{box['category']}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def overlay_seg_mask(image: np.ndarray, seg_color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    seg = cv2.resize(seg_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image, 1 - alpha, seg, alpha, 0)
