"""Visualize ORB keypoints on baseline, distorted, and enhanced images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.apply_distortions import (
    _PREVIEW_DISTORTION_NAMES,
    _PREVIEW_ENHANCEMENT_NAMES,
    _preview_distorted_label,
    list_images,
)
from src.distortions import apply_distortion, default_levels
from src.enhancements import enhance_for_distortion
from src.evaluate import draw_orb_keypoints, draw_orb_matches_panel, orb_matching_ratio
from src.paths import DISTORTION_TYPES, FIGURES_DIR, ensure_output_dirs


def _level_seed(base_seed: int, distortion: str, level: float | int) -> int:
    import hashlib

    key = f"{base_seed}:{distortion}:{level}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def save_orb_keypoints_preview(
    split: str,
    preview_seed: int,
    output_path: Path,
    max_features: int = 500,
) -> None:
    """3x3 grid: rows=distortions, cols=baseline / distorted / enhanced with ORB keypoints."""
    images = list_images(split, limit=1, seed=preview_seed)
    if not images:
        return

    image_path = images[0]
    clean = cv2.imread(str(image_path))
    if clean is None:
        return

    column_titles = ["Baseline (clean)", "Distorted", "Enhanced"]
    n_rows = len(DISTORTION_TYPES)

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    _, n_clean_kp = draw_orb_keypoints(clean, max_features=max_features)

    for row, distortion in enumerate(DISTORTION_TYPES):
        level = default_levels(distortion)[-1]
        rng = np.random.default_rng(_level_seed(preview_seed, distortion, level))
        distorted = apply_distortion(clean, distortion, level, rng=rng)
        enhanced = enhance_for_distortion(distorted, distortion)

        panels = [
            (clean, "baseline"),
            (distorted, "distorted"),
            (enhanced, "enhanced"),
        ]

        for col, (img, kind) in enumerate(panels):
            ax = axes[row, col]
            vis, n_kp = draw_orb_keypoints(img, max_features=max_features)
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

            if kind == "baseline":
                caption = f"{n_kp} keypoints\n(reference)"
            elif kind == "distorted":
                m = orb_matching_ratio(clean, img, max_features=max_features)
                caption = (
                    f"{_preview_distorted_label(distortion, level, clean, img)}\n"
                    f"{n_kp} keypoints | "
                    f"match ratio {m['matching_ratio']:.2f} "
                    f"({m['num_good_matches']}/{m['num_keypoints_reference']})"
                )
            else:
                m = orb_matching_ratio(clean, img, max_features=max_features)
                caption = (
                    f"{_PREVIEW_ENHANCEMENT_NAMES[distortion]}\n"
                    f"{n_kp} keypoints | "
                    f"match ratio {m['matching_ratio']:.2f} "
                    f"({m['num_good_matches']}/{m['num_keypoints_reference']})"
                )

            ax.text(
                0.5,
                -0.05,
                caption,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                linespacing=1.35,
            )

    fig.suptitle(
        f"ORB keypoints — {image_path.name}  (reference: {n_clean_kp} keypoints on clean)",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_orb_matches_preview(
    split: str,
    preview_seed: int,
    output_path: Path,
    max_features: int = 500,
) -> None:
    """One row per distortion: clean|query match lines for distorted and enhanced."""
    images = list_images(split, limit=1, seed=preview_seed)
    if not images:
        return

    image_path = images[0]
    clean = cv2.imread(str(image_path))
    if clean is None:
        return

    n_rows = len(DISTORTION_TYPES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, distortion in enumerate(DISTORTION_TYPES):
        level = default_levels(distortion)[-1]
        rng = np.random.default_rng(_level_seed(preview_seed, distortion, level))
        distorted = apply_distortion(clean, distortion, level, rng=rng)
        enhanced = enhance_for_distortion(distorted, distortion)

        for col, (query, title) in enumerate(
            [
                (distorted, "Distorted"),
                (enhanced, "Enhanced"),
            ]
        ):
            vis, metrics = draw_orb_matches_panel(clean, query, max_features=max_features)
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{title}  (clean | query, green=matches)", fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    _PREVIEW_DISTORTION_NAMES[distortion],
                    rotation=90,
                    labelpad=44,
                    fontsize=11,
                    fontweight="bold",
                )
            ax.text(
                0.5,
                -0.03,
                f"match ratio {metrics['matching_ratio']:.2f}  "
                f"({metrics['num_good_matches']}/{metrics['num_keypoints_reference']})",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

    fig.suptitle(f"ORB matches — {image_path.name}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ORB keypoints on distorted/enhanced images.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--preview-seed", type=int, default=99)
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "orb_keypoints_preview_train.png",
    )
    parser.add_argument(
        "--matches-output",
        type=Path,
        default=FIGURES_DIR / "orb_matches_preview_train.png",
    )
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--matches", action="store_true", help="Also save match-lines figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    save_orb_keypoints_preview(
        args.split,
        args.preview_seed,
        args.output,
        max_features=args.max_features,
    )
    print(f"Saved keypoints preview: {args.output}")

    if args.matches:
        save_orb_matches_preview(
            args.split,
            args.preview_seed,
            args.matches_output,
            max_features=args.max_features,
        )
        print(f"Saved matches preview: {args.matches_output}")


if __name__ == "__main__":
    main()
