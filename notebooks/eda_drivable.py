"""EDA: visualize BDD100K drivable-area GT — one overlay column per category."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = PROJECT_ROOT / "data" / "bdd100K_images_10k" / "10k"
DRIVABLE_COLOR_ROOT = PROJECT_ROOT / "bdd100k_drivable_maps" / "color_labels"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "eda_drivable_samples.png"

# BDD100K drivable color labels (BGR)
DRIVABLE_CATEGORIES: list[tuple[str, tuple[int, int, int]]] = [
    ("direct drivable", (0, 0, 255)),       # red in RGB
    ("alternative drivable", (255, 0, 0)),  # blue in RGB
    ("non-drivable", (0, 0, 0)),            # background / not drivable
]


def drivable_color_path(split: str, stem: str) -> Path:
    return DRIVABLE_COLOR_ROOT / split / f"{stem}_drivable_color.png"


def collect_samples(split: str, limit: int) -> list[Path]:
    image_dir = IMAGES_ROOT / split
    images = sorted(image_dir.glob("*.jpg"))
    images = [p for p in images if drivable_color_path(split, p.stem).exists()]
    return images[:limit] if limit else images


def overlay_single_category(
    image: np.ndarray,
    mask_bgr: np.ndarray,
    category_bgr: tuple[int, int, int],
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend only pixels belonging to one drivable category onto the image."""
    mask = cv2.resize(
        mask_bgr,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    cat_mask = np.all(mask == category_bgr, axis=2)

    colored = np.zeros_like(image)
    colored[cat_mask] = category_bgr

    vis = image.copy()
    if not cat_mask.any():
        return vis

    blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    vis[cat_mask] = blended[cat_mask]
    return vis


def build_panel(image_path: Path, split: str) -> list[np.ndarray] | None:
    drivable_path = drivable_color_path(split, image_path.stem)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: could not read {image_path}")
        return None

    mask = cv2.imread(str(drivable_path))
    if mask is None:
        print(f"Warning: could not read {drivable_path}")
        return None

    panel = [image]
    for name, color_bgr in DRIVABLE_CATEGORIES:
        panel.append(overlay_single_category(image, mask, color_bgr))
    return panel


def print_dataset_summary() -> None:
    print("=== BDD100K drivable-area summary ===")
    for split in ("train", "val", "test"):
        n_images = len(list((IMAGES_ROOT / split).glob("*.jpg")))
        drivable_dir = DRIVABLE_COLOR_ROOT / split
        n_drivable = len(list(drivable_dir.glob("*.png"))) if drivable_dir.exists() else 0
        paired = sum(
            1
            for p in (IMAGES_ROOT / split).glob("*.jpg")
            if drivable_color_path(split, p.stem).exists()
        )
        print(f"{split:5s}  images={n_images:5d}  drivable_masks={n_drivable:6d}  paired={paired:5d}")


def save_grid(samples: list[Path], split: str, output_path: Path) -> None:
    n = len(samples)
    n_cols = 1 + len(DRIVABLE_CATEGORIES)
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Original"] + [name for name, _ in DRIVABLE_CATEGORIES]

    for row, image_path in enumerate(samples):
        panel = build_panel(image_path, split)
        if panel is None:
            continue

        for col, vis in enumerate(panel):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(column_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(image_path.stem, rotation=90, labelpad=40, fontsize=8)

    fig.suptitle(
        f"BDD100K drivable-area GT — {split} split ({n} samples, one column per category)",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BDD100K drivable-area GT per category.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_dataset_summary()

    candidates = collect_samples(args.split, limit=0)
    if not candidates:
        raise SystemExit(f"No images with drivable masks found for split '{args.split}'.")

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(candidates))
    samples = rng.sample(candidates, n)

    print(f"\nSelected {n} samples from '{args.split}':")
    for path in samples:
        print(f"  - {path.name}")

    save_grid(samples, args.split, args.output)


if __name__ == "__main__":
    main()
