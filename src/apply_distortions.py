"""Batch-apply distortions (and optional enhancements) to dataset images."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.distortions import apply_distortion, compute_snr_db, default_levels, level_tag
from src.enhancements import enhance_for_distortion
from src.paths import DISTORTED_DIR, ENHANCED_DIR, FIGURES_DIR, IMAGES_ROOT, DISTORTION_TYPES, ensure_output_dirs

# Labels used only in distortion_preview_*.png
_PREVIEW_DISTORTION_NAMES = {
    "noise": "Gaussian Noise",
    "low_light": "Low Light",
    "jpeg": "JPEG Compression",
}
_PREVIEW_ENHANCEMENT_NAMES = {
    "noise": "NLM Denoising",
    "low_light": "CLAHE",
    "jpeg": "Bilateral + Interpolation",
}


def _preview_distorted_label(
    distortion: str,
    level: float | int,
    clean: np.ndarray,
    distorted: np.ndarray,
) -> str:
    measured_snr = compute_snr_db(clean, distorted)
    if distortion == "noise":
        return f"Target SNR: {level} dB\nMeasured SNR: {measured_snr:.1f} dB"
    if distortion == "low_light":
        return f"Gamma: {level:g}\nSNR vs clean: {measured_snr:.1f} dB"
    if distortion == "jpeg":
        return f"JPEG quality: {int(level)}"
    return str(level)


def list_images(split: str, limit: int | None = None, seed: int = 42) -> list[Path]:
    images = sorted((IMAGES_ROOT / split).glob("*.jpg"))
    if limit is not None and limit < len(images):
        rng = random.Random(seed)
        images = rng.sample(images, limit)
        images.sort()
    return images


def distorted_path(distortion: str, level: float | int, split: str, image_name: str) -> Path:
    return DISTORTED_DIR / distortion / level_tag(distortion, level) / split / image_name


def enhanced_path(distortion: str, level: float | int, split: str, image_name: str) -> Path:
    return ENHANCED_DIR / distortion / level_tag(distortion, level) / split / image_name


def process_images(
    split: str,
    distortions: list[str],
    levels: dict[str, list[float | int]] | None,
    num_images: int | None,
    seed: int,
    also_enhance: bool,
    overwrite: bool,
) -> dict:
    ensure_output_dirs()
    rng = np.random.default_rng(seed)
    images = list_images(split, limit=num_images, seed=seed)

    manifest: dict = {
        "split": split,
        "seed": seed,
        "num_images": len(images),
        "distortions": {},
    }

    for distortion in distortions:
        level_list = levels[distortion] if levels else default_levels(distortion)
        manifest["distortions"][distortion] = {"levels": level_list, "images": {}}

        for level in level_list:
            tag = level_tag(distortion, level)
            manifest["distortions"][distortion]["images"][tag] = []

            for image_path in tqdm(images, desc=f"{distortion}/{tag}"):
                clean = cv2.imread(str(image_path))
                if clean is None:
                    continue

                out_dist = distorted_path(distortion, level, split, image_path.name)
                if not overwrite and out_dist.exists():
                    distorted = cv2.imread(str(out_dist))
                else:
                    distorted = apply_distortion(clean, distortion, level, rng=rng)
                    out_dist.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_dist), distorted)

                record = {"file": image_path.name, "distorted_path": str(out_dist)}

                if distortion == "noise":
                    record["measured_snr_db"] = round(compute_snr_db(clean, distorted), 2)
                elif distortion == "low_light":
                    record["measured_snr_db"] = round(compute_snr_db(clean, distorted), 2)

                if also_enhance:
                    out_enh = enhanced_path(distortion, level, split, image_path.name)
                    if not overwrite and out_enh.exists():
                        record["enhanced_path"] = str(out_enh)
                    else:
                        enhanced = enhance_for_distortion(distorted, distortion)
                        out_enh.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_enh), enhanced)
                        record["enhanced_path"] = str(out_enh)

                manifest["distortions"][distortion]["images"][tag].append(record)

    manifest_path = DISTORTED_DIR / f"manifest_{split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def save_preview(split: str, preview_seed: int, output_path: Path) -> None:
    """Save a grid with one row per distortion: Original | Distorted | Enhanced."""
    images = list_images(split, limit=1, seed=preview_seed)
    if not images:
        return

    image_path = images[0]
    clean = cv2.imread(str(image_path))
    if clean is None:
        return

    rng = np.random.default_rng(preview_seed)
    n_rows = len(DISTORTION_TYPES)
    column_titles = ["Original", "Distorted", "Enhanced"]

    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, distortion in enumerate(DISTORTION_TYPES):
        level = default_levels(distortion)[-1]  # strongest distortion
        distorted = apply_distortion(clean, distortion, level, rng=rng)
        enhanced = enhance_for_distortion(distorted, distortion)
        row_images = [clean, distorted, enhanced]

        for col, vis in enumerate(row_images):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(column_titles[col], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    _PREVIEW_DISTORTION_NAMES[distortion],
                    rotation=90,
                    labelpad=44,
                    fontsize=11,
                    fontweight="bold",
                )
                caption = "Clean image"
            elif col == 1:
                caption = _preview_distorted_label(distortion, level, clean, distorted)
            else:
                caption = _PREVIEW_ENHANCEMENT_NAMES[distortion]

            ax.text(
                0.5,
                -0.05,
                caption,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9.5,
                linespacing=1.4,
            )

    fig.suptitle(f"Distortion preview — {image_path.name}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _level_column_title(distortion: str, level: float | int) -> str:
    if distortion == "noise":
        return f"SNR {int(level)} dB"
    if distortion == "low_light":
        return f"γ = {level:g}"
    if distortion == "jpeg":
        return f"Q = {int(level)}"
    return str(level)


def save_intensity_preview(split: str, preview_seed: int, output_path: Path) -> None:
    """Grid: one row per distortion, columns = clean + all intensity levels."""
    images = list_images(split, limit=1, seed=preview_seed)
    if not images:
        return

    image_path = images[0]
    clean = cv2.imread(str(image_path))
    if clean is None:
        return

    rng = np.random.default_rng(preview_seed)
    n_rows = len(DISTORTION_TYPES)
    n_cols = 5  # clean + 4 levels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, distortion in enumerate(DISTORTION_TYPES):
        levels = default_levels(distortion)
        panels = [clean] + [
            apply_distortion(clean, distortion, level, rng=rng) for level in levels
        ]
        titles = ["Clean"] + [_level_column_title(distortion, lv) for lv in levels]

        for col, (vis, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Distortion intensity sweep - {image_path.name}",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.09)
    for row, distortion in enumerate(DISTORTION_TYPES):
        pos = axes[row, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(
            0.025,
            y_center,
            _PREVIEW_DISTORTION_NAMES[distortion],
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_enhancement_preview(split: str, preview_seed: int, output_path: Path) -> None:
    """Grid: one row per distortion — distorted (strongest) vs enhanced side by side."""
    images = list_images(split, limit=1, seed=preview_seed)
    if not images:
        return

    image_path = images[0]
    clean = cv2.imread(str(image_path))
    if clean is None:
        return

    rng = np.random.default_rng(preview_seed)
    n_rows = len(DISTORTION_TYPES)

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, distortion in enumerate(DISTORTION_TYPES):
        level = default_levels(distortion)[-1]
        distorted = apply_distortion(clean, distortion, level, rng=rng)
        enhanced = enhance_for_distortion(distorted, distortion)

        for col, (vis, title) in enumerate(
            [
                (distorted, "Distorted"),
                (enhanced, "After enhancement"),
            ]
        ):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    _PREVIEW_DISTORTION_NAMES[distortion],
                    rotation=90,
                    labelpad=44,
                    fontsize=11,
                    fontweight="bold",
                )
                caption = _preview_distorted_label(distortion, level, clean, distorted)
            else:
                caption = _PREVIEW_ENHANCEMENT_NAMES[distortion]

            ax.text(
                0.5,
                -0.06,
                caption,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9.5,
                linespacing=1.4,
            )

    fig.suptitle(f"Enhancement recovery — {image_path.name}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply distortions to BDD100K images.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-images", type=int, default=50, help="Number of images (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distortions",
        nargs="+",
        default=list(DISTORTION_TYPES),
        choices=DISTORTION_TYPES,
    )
    parser.add_argument("--also-enhance", action="store_true", help="Also save enhanced images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--preview", action="store_true", help="Save a preview figure")
    parser.add_argument(
        "--preview-seed",
        type=int,
        default=99,
        help="Seed for picking the preview image (default: 99)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = process_images(
        split=args.split,
        distortions=args.distortions,
        levels=None,
        num_images=args.num_images,
        seed=args.seed,
        also_enhance=args.also_enhance,
        overwrite=args.overwrite,
    )

    print(f"\nProcessed {manifest['num_images']} images from split '{args.split}'")
    print(f"Manifest: {DISTORTED_DIR / f'manifest_{args.split}.json'}")

    if args.preview:
        save_preview(
            args.split,
            args.preview_seed,
            FIGURES_DIR / f"distortion_preview_{args.split}.png",
        )
        save_intensity_preview(
            args.split,
            args.preview_seed,
            FIGURES_DIR / f"distortion_intensity_{args.split}.png",
        )
        save_enhancement_preview(
            args.split,
            args.preview_seed,
            FIGURES_DIR / f"enhancement_preview_{args.split}.png",
        )
        print(f"Previews saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
