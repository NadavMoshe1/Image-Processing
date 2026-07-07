"""Exploratory data analysis: visualize BDD100K samples with annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = PROJECT_ROOT / "data" / "bdd100K_images_10k" / "10k"
LABELS_ROOT = PROJECT_ROOT / "data" / "bdd100k_label" / "100k"
SEG_COLOR_ROOT = PROJECT_ROOT / "data" / "bdd100k_seg_maps" / "color_labels"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "eda_samples.png"

BOX_COLORS = {
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
    """Map image stem to JSON label path (labels may live in any split folder)."""
    index: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for label_path in (LABELS_ROOT / split).glob("*.json"):
            index[label_path.stem] = label_path
    return index


def resolve_label_path(stem: str, label_index: dict[str, Path]) -> Path | None:
    return label_index.get(stem)


def load_label_boxes(label_path: Path | None) -> list[dict]:
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
                    "x1": int(box["x1"]),
                    "y1": int(box["y1"]),
                    "x2": int(box["x2"]),
                    "y2": int(box["y2"]),
                }
            )
    return boxes


def draw_boxes(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    vis = image.copy()
    for box in boxes:
        color = BOX_COLORS.get(box["category"], (255, 255, 255))
        cv2.rectangle(vis, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        cv2.putText(
            vis,
            box["category"],
            (box["x1"], max(box["y1"] - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def overlay_segmentation(image: np.ndarray, seg_color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    seg = cv2.resize(seg_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image, 1 - alpha, seg, alpha, 0)


def draw_orb_keypoints(image: np.ndarray, max_features: int = 500) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints = orb.detect(gray, None)
    return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)


def seg_color_path(split: str, stem: str) -> Path:
    return SEG_COLOR_ROOT / split / f"{stem}_train_color.png"


def collect_samples(
    split: str,
    require_seg: bool,
    require_boxes: bool,
    label_index: dict[str, Path],
    limit: int,
) -> list[Path]:
    image_dir = IMAGES_ROOT / split
    images = sorted(image_dir.glob("*.jpg"))
    if require_seg:
        images = [p for p in images if seg_color_path(split, p.stem).exists()]
    if require_boxes:
        images = [p for p in images if load_label_boxes(resolve_label_path(p.stem, label_index))]
    return images[:limit] if limit else images


def print_dataset_summary(label_index: dict[str, Path]) -> None:
    print("=== BDD100K dataset summary ===")
    for split in ("train", "val", "test"):
        n_images = len(list((IMAGES_ROOT / split).glob("*.jpg")))
        n_labels = len(list((LABELS_ROOT / split).glob("*.json")))
        seg_dir = SEG_COLOR_ROOT / split
        n_seg = len(list(seg_dir.glob("*.png"))) if seg_dir.exists() else 0
        print(f"{split:5s}  images={n_images:5d}  json_labels={n_labels:6d}  seg_masks={n_seg:5d}")

    paired = 0
    missing_label = 0
    missing_seg = 0
    missing_boxes = 0
    for split in ("train", "val"):
        for image_path in (IMAGES_ROOT / split).glob("*.jpg"):
            label_path = resolve_label_path(image_path.stem, label_index)
            if label_path is None:
                missing_label += 1
                continue
            if not seg_color_path(split, image_path.stem).exists():
                missing_seg += 1
                continue
            if not load_label_boxes(label_path):
                missing_boxes += 1
                continue
            paired += 1
    print(f"train+val with image + json + seg + box2d: {paired}")
    print(f"train+val missing json label:             {missing_label}")
    print(f"train+val missing seg mask:               {missing_seg}")
    print(f"train+val missing box2d annotations:      {missing_boxes}")


def build_panel(
    image_path: Path,
    split: str,
    label_index: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    label_path = resolve_label_path(image_path.stem, label_index)
    seg_path = seg_color_path(split, image_path.stem)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: could not read {image_path}")
        return None

    boxes = load_label_boxes(label_path)
    if not boxes:
        print(f"Warning: no box2d labels for {image_path.name}")
    bbox_vis = draw_boxes(image, boxes)

    if seg_path.exists():
        seg_color = cv2.imread(str(seg_path))
        seg_vis = overlay_segmentation(image, seg_color) if seg_color is not None else image.copy()
    else:
        seg_vis = image.copy()

    orb_vis = draw_orb_keypoints(image)
    return image, bbox_vis, seg_vis, orb_vis


def save_grid(
    samples: list[Path],
    split: str,
    output_path: Path,
    label_index: dict[str, Path],
) -> None:
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Original", "Detection (box2d)", "Segmentation overlay", "ORB keypoints"]

    for row, image_path in enumerate(samples):
        panel = build_panel(image_path, split, label_index)
        if panel is None:
            continue

        for col, vis in enumerate(panel):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(column_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(image_path.stem, rotation=90, labelpad=40, fontsize=9)

    fig.suptitle(
        f"BDD100K EDA: {split} split ({n} sample{'s' if n != 1 else ''})",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BDD100K samples with annotations.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_index = build_label_index()
    print_dataset_summary(label_index)

    candidates = collect_samples(
        args.split,
        require_seg=(args.split != "test"),
        require_boxes=True,
        label_index=label_index,
        limit=0,
    )
    if not candidates:
        raise SystemExit(
            f"No images with box2d labels found for split '{args.split}'. "
            "Try --split train (val seg subset has no detection JSON in this download)."
        )

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(candidates))
    samples = rng.sample(candidates, n)

    print(f"\nSelected {n} samples from '{args.split}':")
    for path in samples:
        label_path = resolve_label_path(path.stem, label_index)
        n_boxes = len(load_label_boxes(label_path))
        print(f"  - {path.name}  ({n_boxes} boxes)")

    save_grid(samples, args.split, args.output, label_index)


if __name__ == "__main__":
    main()
