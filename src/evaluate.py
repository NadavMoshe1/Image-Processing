"""Shared metrics and plotting utilities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.paths import DISTORTION_TYPES, METRICS_DIR, ensure_output_dirs
from src.enhancements import enhancement_label


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
        ft_data = results.get("finetuned", {}).get(distortion, {})
        levels = sorted_level_keys(distortion, list(dist_data.keys()))

        x_vals = [float(l) if distortion != "jpeg" else int(l) for l in levels]
        y_dist = [dist_data[l][metric_key] for l in levels]
        y_enh = [enh_data[l][metric_key] for l in levels if l in enh_data]
        y_ft = [ft_data[l][metric_key] for l in levels if l in ft_data]

        ax.plot(x_vals, y_dist, marker="o", label="Distorted")
        if y_enh:
            ax.plot(x_vals[: len(y_enh)], y_enh, marker="s", linestyle="--", label="Enhanced")
        if y_ft:
            ax.plot(x_vals[: len(y_ft)], y_ft, marker="^", linestyle="-.", label="Fine-tuned")
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


def load_detection_finetune_batch_results(
    summary_path: Path | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Load YOLO fine-tune batch summary and group metrics by distortion type."""
    from src.distortions import default_levels, level_tag

    if summary_path is None:
        summary_path = METRICS_DIR / "finetune_batch_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Batch summary not found: {summary_path}")

    batch = json.loads(summary_path.read_text(encoding="utf-8"))
    by_job = {
        row["job"]: row
        for row in batch.get("results", [])
        if not row.get("skipped") and "mean_recall_pretrained" in row
    }

    grouped: dict[str, list[dict]] = {distortion: [] for distortion in DISTORTION_TYPES}
    for distortion in DISTORTION_TYPES:
        for level in default_levels(distortion):
            tag = level_tag(distortion, level)
            job_name = f"{distortion}/{tag}"
            row = by_job.get(job_name)
            if row is None:
                continue
            grouped[distortion].append(
                {
                    "level": float(level) if distortion != "jpeg" else int(level),
                    "tag": tag,
                    "job": job_name,
                    "pretrained": float(row["mean_recall_pretrained"]),
                    "enhanced": float(row["mean_recall_enhanced"]),
                    "finetuned": float(row["mean_recall_finetuned"]),
                }
            )
        grouped[distortion].sort(
            key=lambda item: item["level"],
            reverse=(distortion != "low_light"),
        )
    return grouped, batch


def load_detection_clean_baseline_recall(split: str = "train") -> float | None:
    """Mean recall @ IoU 0.5 on clean images from the robustness evaluation set."""
    path = METRICS_DIR / f"detection_baseline_{split}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    recall = data.get("mean_recall")
    return float(recall) if recall is not None else None


def save_detection_finetune_summary_recall_plot(
    grouped: dict[str, list[dict]],
    output_path: Path,
    *,
    title: str = "YOLO fine-tuning — recall vs distortion intensity",
    subtitle: str | None = None,
    baseline_recall: float | None = None,
    ylabel: str = "Recall @ IoU 0.5",
    baseline_label: str | None = None,
    ylim_max: float = 0.55,
) -> Path:
    """Line plot of pretrained / enhanced / fine-tuned recall across all FT levels."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xlabels = {
        "noise": "SNR (dB)",
        "low_light": "Gamma",
        "jpeg": "JPEG quality",
    }
    series_style = {
        "pretrained": {"marker": "o", "linestyle": "-", "color": "#95a5a6", "label": "Pretrained"},
        "finetuned": {"marker": "^", "linestyle": "-", "color": "#2ecc71", "label": "Fine-tuned"},
    }

    fig, axes = plt.subplots(1, len(DISTORTION_TYPES), figsize=(5 * len(DISTORTION_TYPES), 4.8))
    if len(DISTORTION_TYPES) == 1:
        axes = [axes]

    for ax, distortion in zip(axes, DISTORTION_TYPES):
        rows = grouped.get(distortion, [])
        if not rows:
            ax.set_visible(False)
            continue

        x_vals = [row["level"] for row in rows]
        for key in ("pretrained", "enhanced", "finetuned"):
            if key == "enhanced":
                style = {
                    "marker": "s",
                    "linestyle": "--",
                    "color": "#f39c12",
                    "label": enhancement_label(distortion),
                }
            else:
                style = series_style[key]
            ax.plot(
                x_vals,
                [row[key] for row in rows],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2,
                markersize=7,
                label=style["label"],
            )

        ax.set_xlabel(xlabels[distortion])
        ax.set_ylabel(ylabel)
        ax.set_title(distortion.replace("_", " ").title())
        y_max = max(
            (row[key] for row in rows for key in ("pretrained", "enhanced", "finetuned")),
            default=0.0,
        )
        if baseline_recall is not None:
            bl_label = baseline_label or f"Clean baseline ({baseline_recall:.2f})"
            ax.axhline(
                baseline_recall,
                color="#3498db",
                linestyle=":",
                linewidth=2,
                label=bl_label,
            )
            y_max = max(y_max, baseline_recall)
        ax.set_ylim(0, max(ylim_max, y_max * 1.12))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    if subtitle:
        fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
    else:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_detection_finetune_summary_gain_plot(
    grouped: dict[str, list[dict]],
    output_path: Path,
    *,
    title: str = "YOLO fine-tuning — recall gain over pretrained",
    ylabel: str = "Recall gain (fine-tuned − pretrained)",
) -> Path:
    """Bar chart of fine-tuned minus pretrained recall for every distortion level."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    gains: list[float] = []
    colors: list[str] = []
    palette = {"noise": "#3498db", "low_light": "#9b59b6", "jpeg": "#e67e22"}

    for distortion in DISTORTION_TYPES:
        for row in grouped.get(distortion, []):
            labels.append(row["tag"].replace("_", " "))
            gain = row["finetuned"] - row["pretrained"]
            gains.append(gain)
            colors.append(palette[distortion])

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(labels)), 5))
    bars = ax.bar(labels, gains, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    for bar, gain in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.008 if gain >= 0 else -0.018),
            f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}",
            ha="center",
            va="bottom" if gain >= 0 else "top",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_detection_finetune_summary_table_plot(
    grouped: dict[str, list[dict]],
    output_path: Path,
    *,
    title: str = "YOLO fine-tuning — recall by condition",
    baseline_recall: float | None = None,
    cbar_label: str = "Recall @ IoU 0.5",
    vmax: float = 0.5,
) -> Path:
    """Heatmap-style table of pretrained / enhanced / fine-tuned recall."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[str | float]] = []
    for distortion in DISTORTION_TYPES:
        for item in grouped.get(distortion, []):
            rows.append(
                [
                    item["tag"],
                    item["pretrained"],
                    item["enhanced"],
                    item["finetuned"],
                    item["finetuned"] - item["pretrained"],
                ]
            )

    if not rows:
        raise ValueError("No fine-tune batch results to plot.")

    labels = [row[0] for row in rows]
    col_names = ["Pretrained", "Enhanced", "Fine-tuned"]
    values = np.array([[row[1], row[2], row[3]] for row in rows], dtype=float)
    if baseline_recall is not None:
        col_names = ["Clean baseline"] + col_names
        baseline_col = np.full((len(rows), 1), baseline_recall, dtype=float)
        values = np.hstack([baseline_col, values])

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(labels))))
    im = ax.imshow(values, aspect="auto", cmap="YlGn", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#222222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_segmentation_finetune_batch_results(
    summary_path: Path | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Load SegFormer fine-tune batch summary; fill gaps from per-job eval JSON files."""
    from src.distortions import default_levels, level_tag

    if summary_path is None:
        summary_path = METRICS_DIR / "seg_finetune_batch_summary.json"

    batch: dict = {}
    by_job: dict[str, dict] = {}
    if summary_path.exists():
        batch = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in batch.get("results", []):
            if not row.get("skipped") and "mean_miou_pretrained" in row:
                by_job[row["job"]] = row

    grouped: dict[str, list[dict]] = {distortion: [] for distortion in DISTORTION_TYPES}
    for distortion in DISTORTION_TYPES:
        for level in default_levels(distortion):
            tag = level_tag(distortion, level)
            job_name = f"{distortion}/{tag}"
            row = by_job.get(job_name)
            if row is None:
                eval_path = METRICS_DIR / f"segmentation_finetune_eval_{distortion}_{tag}.json"
                if eval_path.exists():
                    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
                    row = {
                        "mean_miou_pretrained": eval_data["mean_miou_pretrained"],
                        "mean_miou_enhanced": eval_data["mean_miou_enhanced"],
                        "mean_miou_finetuned": eval_data["mean_miou_finetuned"],
                    }
            if row is None:
                continue
            grouped[distortion].append(
                {
                    "level": float(level) if distortion != "jpeg" else int(level),
                    "tag": tag,
                    "job": job_name,
                    "pretrained": float(row["mean_miou_pretrained"]),
                    "enhanced": float(row["mean_miou_enhanced"]),
                    "finetuned": float(row["mean_miou_finetuned"]),
                }
            )
        grouped[distortion].sort(
            key=lambda item: item["level"],
            reverse=(distortion != "low_light"),
        )

    if not batch:
        batch = {"num_train": 500, "num_val": 100, "epochs": 30}
    return grouped, batch


def load_segmentation_clean_baseline_miou(split: str = "train") -> float | None:
    """Mean mIoU on clean images from the robustness evaluation set."""
    path = METRICS_DIR / f"segmentation_baseline_{split}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    miou = data.get("mean_miou")
    return float(miou) if miou is not None else None


def save_segmentation_finetune_summary_recall_plot(
    grouped: dict[str, list[dict]],
    output_path: Path,
    *,
    subtitle: str | None = None,
    baseline_miou: float | None = None,
) -> Path:
    baseline_label = f"Clean baseline ({baseline_miou:.2f})" if baseline_miou is not None else None
    return save_detection_finetune_summary_recall_plot(
        grouped,
        output_path,
        title="SegFormer fine-tuning — mIoU vs distortion intensity",
        subtitle=subtitle,
        baseline_recall=baseline_miou,
        ylabel="mIoU",
        baseline_label=baseline_label,
        ylim_max=0.58,
    )


def save_segmentation_finetune_summary_gain_plot(
    grouped: dict[str, list[dict]],
    output_path: Path,
) -> Path:
    return save_detection_finetune_summary_gain_plot(
        grouped,
        output_path,
        title="SegFormer fine-tuning — mIoU gain over pretrained",
        ylabel="mIoU gain (fine-tuned − pretrained)",
    )


def save_segmentation_finetune_summary_table_plot(
    grouped: dict[str, list[dict]],
    output_path: Path,
    *,
    baseline_miou: float | None = None,
) -> Path:
    return save_detection_finetune_summary_table_plot(
        grouped,
        output_path,
        title="SegFormer fine-tuning — mIoU by condition",
        baseline_recall=baseline_miou,
        cbar_label="mIoU",
        vmax=0.55,
    )


def save_comparison_bars(
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    output_path: Path,
    colors: list[str] | None = None,
    baseline: float | None = None,
    baseline_label: str = "Clean baseline",
) -> Path:
    """Save a bar chart comparing metric values across conditions."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(labels)), 5))
    bar_colors = colors or ["#3498db", "#2ecc71"][: len(labels)]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    y_top = max(max(values), baseline or 0.0)
    ax.set_ylim(0, max(y_top * 1.15, 0.05))
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#3498db",
            linestyle=":",
            linewidth=2,
            label=f"{baseline_label} ({baseline:.3f})",
        )
        ax.legend(fontsize=9, loc="upper right")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_finetune_training_curves(results_csv: Path, output_path: Path, title: str) -> Path:
    """Plot YOLO training losses and validation metrics from Ultralytics results.csv."""
    import csv

    ensure_output_dirs()
    epochs: list[int] = []
    box_loss: list[float] = []
    cls_loss: list[float] = []
    map50: list[float] = []
    recall: list[float] = []

    with results_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            box_loss.append(float(row["train/box_loss"]))
            cls_loss.append(float(row["train/cls_loss"]))
            map50.append(float(row["metrics/mAP50(B)"]))
            recall.append(float(row["metrics/recall(B)"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, box_loss, label="box loss", marker="o", markersize=3)
    axes[0].plot(epochs, cls_loss, label="cls loss", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training losses")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, map50, label="mAP@50", marker="o", markersize=3)
    axes[1].plot(epochs, recall, label="recall", marker="s", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation metric")
    axes[1].set_title("Validation metrics (noisy val set)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_segformer_training_curves(log_history: list[dict], output_path: Path, title: str) -> Path:
    """Plot SegFormer training loss and validation mIoU from HuggingFace Trainer log history."""
    ensure_output_dirs()
    train_epochs: list[float] = []
    train_loss: list[float] = []
    eval_epochs: list[float] = []
    eval_miou: list[float] = []
    eval_loss: list[float] = []

    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_epochs.append(float(epoch))
            train_loss.append(float(entry["loss"]))
        if "eval_miou" in entry:
            eval_epochs.append(float(epoch))
            eval_miou.append(float(entry["eval_miou"]))
            if "eval_loss" in entry:
                eval_loss.append(float(entry["eval_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if train_epochs:
        axes[0].plot(train_epochs, train_loss, marker="o", markersize=3, label="train loss")
    if eval_epochs and eval_loss:
        axes[0].plot(eval_epochs, eval_loss, marker="s", markersize=3, label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / validation loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if eval_epochs and eval_miou:
        axes[1].plot(eval_epochs, eval_miou, marker="o", markersize=3, color="#2ecc71")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mIoU")
        axes[1].set_title("Validation mIoU (distorted val set)")
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, max(max(eval_miou) * 1.15, 0.05))
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No validation mIoU logged", ha="center", va="center")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_per_class_recall_chart(
    per_class_recall: dict[str, float],
    *,
    output_path: Path,
    title: str = "YOLOv8 per-class recall @ IoU 0.5 (clean baseline)",
) -> Path:
    """Horizontal bar chart of detection recall per object class."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(per_class_recall.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.title() for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels))))
    colors = ["#2ecc71" if v >= 0.35 else "#f39c12" if v >= 0.15 else "#e74c3c" for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Recall @ IoU 0.5")
    ax.set_title(title)
    ax.axvline(np.mean(values), color="#3498db", linestyle="--", linewidth=1.2, label="Mean recall")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_per_class_miou_chart(
    per_class_miou: dict[str, float],
    *,
    output_path: Path,
    title: str = "SegFormer per-class IoU (clean baseline)",
    mean_miou: float | None = None,
) -> Path:
    """Horizontal bar chart of segmentation IoU per semantic class."""
    ensure_output_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(per_class_miou.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.title() for k, _ in items]
    values = [v for _, v in items]
    mean_val = mean_miou if mean_miou is not None else float(np.mean(values))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels))))
    colors = ["#2ecc71" if v >= 0.55 else "#f39c12" if v >= 0.25 else "#e74c3c" for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("IoU")
    ax.set_title(title)
    ax.axvline(mean_val, color="#3498db", linestyle="--", linewidth=1.2, label=f"Mean mIoU ({mean_val:.2f})")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                min(val + 0.02, 0.95),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=9,
            )
    ax.legend(loc="lower right", fontsize=9)
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


def new_detection_class_stats() -> dict[str, dict[str, int]]:
    return {}


def update_detection_class_stats(
    class_stats: dict[str, dict[str, int]],
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    iou_thresh: float = 0.5,
) -> None:
    """Accumulate per-class GT counts and matches (greedy, same as baseline)."""
    for box in gt_boxes:
        cat = box["category"]
        class_stats.setdefault(cat, {"gt": 0, "matched": 0})
        class_stats[cat]["gt"] += 1

    matched_gt: set[int] = set()
    for pred in sorted(pred_boxes, key=lambda b: b.get("confidence", 0), reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt or pred["category"] != gt["category"]:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            class_stats[pred["category"]]["matched"] += 1


def per_class_recall_from_stats(class_stats: dict[str, dict[str, int]]) -> dict[str, float]:
    return {
        cat: (v["matched"] / v["gt"] if v["gt"] else 0.0)
        for cat, v in class_stats.items()
    }


def new_segmentation_class_accumulator() -> dict[int, list[float]]:
    return {}


def update_segmentation_class_accumulator(
    acc: dict[int, list[float]],
    per_class_iou: dict[int, float],
) -> None:
    for cls_id, iou in per_class_iou.items():
        acc.setdefault(cls_id, []).append(float(iou))


def per_class_miou_from_accumulator(
    acc: dict[int, list[float]],
    class_names: list[str],
) -> dict[str, float]:
    return {
        (class_names[cls_id] if cls_id < len(class_names) else str(cls_id)): float(np.mean(vals))
        for cls_id, vals in acc.items()
        if vals
    }


def compute_miou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 19) -> dict:
    """Per-class and mean IoU; ignore pixels where gt == 255."""
    pred_mask = np.squeeze(pred_mask)
    gt_mask = np.squeeze(gt_mask)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[:, :, 0]
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
