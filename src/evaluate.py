"""Shared metrics and plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paths import ensure_output_dirs


def save_metric_curve(
    x: list[float] | list[int],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    """Save a single metric-vs-intensity line plot."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_comparison_bars(
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    """Save a bar chart comparing metric values across conditions."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def mean_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> float:
    """Compute mean IoU across classes (for segmentation evaluation)."""
    raise NotImplementedError("Implemented with SegFormer baseline")


def matching_ratio(
    clean_image: np.ndarray,
    other_image: np.ndarray,
    max_features: int = 500,
) -> float:
    """ORB matching ratio between two images (for feature evaluation)."""
    raise NotImplementedError("Implemented in run_orb.py")
