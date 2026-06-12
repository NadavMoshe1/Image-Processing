"""Shared metrics and plotting utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.paths import DISTORTION_TYPES, ensure_output_dirs


def orb_matching_ratio(
    reference_image: np.ndarray,
    query_image: np.ndarray,
    max_features: int = 500,
    ratio_thresh: float = 0.75,
) -> dict:
    """
    Match ORB descriptors from reference (clean) to query (distorted/enhanced).

    Returns matching ratio = good_matches / keypoints_on_reference.
    """
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
    query_kp, query_des = orb.detectAndCompute(query_gray, None)

    n_ref = len(ref_kp) if ref_kp else 0
    if n_ref == 0 or ref_des is None or query_des is None or len(query_kp) == 0:
        return {
            "matching_ratio": 0.0,
            "num_keypoints_reference": n_ref,
            "num_keypoints_query": len(query_kp) if query_kp else 0,
            "num_good_matches": 0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(ref_des, query_des, k=2)

    good_matches = 0
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio_thresh * second.distance:
            good_matches += 1

    return {
        "matching_ratio": good_matches / n_ref,
        "num_keypoints_reference": n_ref,
        "num_keypoints_query": len(query_kp),
        "num_good_matches": good_matches,
    }


def draw_orb_keypoints(
    image: np.ndarray,
    max_features: int = 500,
    color: tuple[int, int, int] = (0, 255, 0),
) -> tuple[np.ndarray, int]:
    """Draw ORB keypoints on image. Returns (visualization, num_keypoints)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints = orb.detect(gray, None)
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return vis, len(keypoints)


def draw_orb_matches_panel(
    reference_image: np.ndarray,
    query_image: np.ndarray,
    max_features: int = 500,
    ratio_thresh: float = 0.75,
) -> tuple[np.ndarray, dict]:
    """Side-by-side clean|query with lines for good ORB matches."""
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
    query_kp, query_des = orb.detectAndCompute(query_gray, None)

    metrics = orb_matching_ratio(reference_image, query_image, max_features, ratio_thresh)

    if (
        not ref_kp
        or not query_kp
        or ref_des is None
        or query_des is None
    ):
        h = max(reference_image.shape[0], query_image.shape[0])
        blank = np.zeros((h, reference_image.shape[1] + query_image.shape[1], 3), dtype=np.uint8)
        blank[: reference_image.shape[0], : reference_image.shape[1]] = reference_image
        blank[: query_image.shape[0], reference_image.shape[1] :] = query_image
        return blank, metrics

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(ref_des, query_des, k=2)
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio_thresh * second.distance:
            good.append(best)

    vis = cv2.drawMatches(
        reference_image,
        ref_kp,
        query_image,
        query_kp,
        good[:80],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis, metrics


def save_metric_curve(
    x: list[float] | list[int],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y2: list[float] | None = None,
    label2: str | None = None,
) -> Path:
    """Save a single metric-vs-intensity line plot."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", label="Distorted")
    if y2 is not None:
        ax.plot(x, y2, marker="s", linestyle="--", label=label2 or "Enhanced")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if y2 is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_robustness_summary_plot(
    results: dict,
    output_path: Path,
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    baseline_metric_key: str | None = None,
) -> Path:
    """Plot metric vs distortion intensity (distorted vs enhanced), ORB-style."""
    from src.robustness import sorted_level_keys

    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xlabels = {
        "noise": "SNR (dB)",
        "low_light": "Gamma",
        "jpeg": "JPEG quality",
    }
    bkey = baseline_metric_key or metric_key
    baseline_val = results.get("baseline", {}).get(bkey, 0.0)

    fig, axes = plt.subplots(1, len(DISTORTION_TYPES), figsize=(5 * len(DISTORTION_TYPES), 4.5))
    if len(DISTORTION_TYPES) == 1:
        axes = [axes]

    for ax, distortion in zip(axes, DISTORTION_TYPES):
        dist_data = results.get("distorted", {}).get(distortion, {})
        enh_data = results.get("enhanced", {}).get(distortion, {})
        levels = sorted_level_keys(distortion, list(dist_data.keys()))

        x_vals = [float(l) if distortion != "jpeg" else int(l) for l in levels]
        y_dist = [dist_data[l][metric_key] for l in levels]
        y_enh = [enh_data[l][metric_key] for l in levels if l in enh_data]

        ax.plot(x_vals, y_dist, marker="o", label="Distorted")
        if y_enh:
            ax.plot(x_vals[: len(y_enh)], y_enh, marker="s", linestyle="--", label="Enhanced")
        ax.axhline(baseline_val, color="green", linestyle=":", linewidth=1.2, label="Clean baseline")
        ax.set_xlabel(xlabels[distortion])
        ax.set_ylabel(ylabel)
        ax.set_title(distortion.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_orb_summary_plot(
    results: dict,
    output_path: Path,
) -> Path:
    """Plot ORB matching ratio vs intensity for all distortions (distorted vs enhanced)."""
    return save_robustness_summary_plot(
        results,
        output_path,
        metric_key="mean_matching_ratio",
        ylabel="Matching ratio",
        title="ORB matching ratio vs distortion intensity",
    )


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


def box_iou(box_a: dict, box_b: dict) -> float:
    """IoU between two boxes dicts with x1,y1,x2,y2."""
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_detection_boxes(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    iou_thresh: float = 0.5,
) -> dict:
    """Greedy match predictions to GT; return recall, precision, mean matched IoU."""
    matched_gt = set()
    matched_pairs: list[tuple[dict, dict, float]] = []

    for pred in sorted(pred_boxes, key=lambda b: b.get("confidence", 0), reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            if pred["category"] != gt["category"]:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            matched_pairs.append((gt_boxes[best_idx], pred, best_iou))

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    n_matched = len(matched_pairs)
    return {
        "recall": n_matched / n_gt if n_gt else 0.0,
        "precision": n_matched / n_pred if n_pred else 0.0,
        "mean_matched_iou": float(np.mean([p[2] for p in matched_pairs])) if matched_pairs else 0.0,
        "num_gt": n_gt,
        "num_pred": n_pred,
        "num_matched": n_matched,
    }


def compute_miou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 19) -> dict:
    """Per-class and mean IoU; ignore pixels where gt == 255."""
    valid = gt_mask != 255
    pred = pred_mask[valid]
    gt = gt_mask[valid]

    ious = {}
    for cls in range(num_classes):
        pred_c = pred == cls
        gt_c = gt == cls
        if not gt_c.any():
            continue
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        ious[cls] = inter / union if union > 0 else 0.0

    return {
        "miou": float(np.mean(list(ious.values()))) if ious else 0.0,
        "per_class_iou": ious,
    }


def mean_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 19) -> float:
    return compute_miou(pred_mask, gt_mask, num_classes)["miou"]
