"""YOLOv8 object detection baseline and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.bdd100k_utils import (
    BDD_CATEGORY_TO_YOLO_ID,
    YOLO_TO_BDD_CATEGORY,
    build_label_index,
    draw_detection_boxes,
    load_detection_boxes,
    resolve_label_path,
    select_paired_images,
)
from src.distortions import apply_distortion, default_levels
from src.enhancements import enhance_for_distortion, enhancement_label
from src.evaluate import (
    box_iou,
    evaluate_detection_boxes,
    load_detection_clean_baseline_recall,
    new_detection_class_stats,
    per_class_recall_from_stats,
    save_comparison_bars,
    save_detection_finetune_summary_gain_plot,
    save_detection_finetune_summary_recall_plot,
    save_detection_finetune_summary_table_plot,
    save_finetune_training_curves,
    save_per_class_recall_chart,
    save_robustness_summary_plot,
    load_detection_finetune_batch_results,
    update_detection_class_stats,
)
from src.paths import (
    BASELINE_DETECTION_DIR,
    DISTORTION_TYPES,
    FIGURES_DIR,
    FINETUNE_DIR,
    METRICS_DIR,
    ensure_output_dirs,
)
from src.robustness import level_seed
from src.yolo_dataset import (
    build_distorted_yolo_dataset,
    dataset_root,
    default_finetune_level,
    load_manifest,
    load_yolo_label_boxes,
)
from src.distortions import level_tag


def yolo_predictions_to_boxes(result, conf_thresh: float = 0.25) -> list[dict]:
    boxes = []
    if result.boxes is None:
        return boxes
    names = result.names
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0])
        yolo_name = names[cls_id]
        bdd_cat = YOLO_TO_BDD_CATEGORY.get(yolo_name, yolo_name)
        if bdd_cat not in BDD_CATEGORY_TO_YOLO_ID:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append(
            {
                "category": bdd_cat,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
            }
        )
    return boxes


def run_baseline(split: str, num_images: int, seed: int, model_name: str) -> dict:
    ensure_output_dirs()
    label_index = build_label_index()
    image_paths = select_paired_images(split, num_images, seed, label_index)
    if not image_paths:
        raise SystemExit(f"No images with detection + seg GT for split '{split}'.")

    model = YOLO(model_name)
    per_image = []
    class_stats: dict[str, dict] = {}

    vis_dir = BASELINE_DETECTION_DIR / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="YOLO detection"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        label_path = resolve_label_path(image_path.stem, label_index)
        gt_boxes = load_detection_boxes(label_path)
        results = model.predict(image, verbose=False)
        pred_boxes = yolo_predictions_to_boxes(results[0])

        metrics = evaluate_detection_boxes(gt_boxes, pred_boxes)
        per_image.append({"image": image_path.name, **metrics, "pred_boxes": pred_boxes, "gt_boxes": gt_boxes})

        gt_vis = draw_detection_boxes(image, gt_boxes)
        pred_vis = draw_detection_boxes(image, pred_boxes)
        combined = np.hstack([gt_vis, pred_vis])
        cv2.imwrite(str(vis_dir / f"{image_path.stem}_gt_pred.jpg"), combined)

        for box in gt_boxes:
            cat = box["category"]
            class_stats.setdefault(cat, {"gt": 0, "matched": 0})
            class_stats[cat]["gt"] += 1

        matched_gt = set()
        for pred in sorted(pred_boxes, key=lambda b: b["confidence"], reverse=True):
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt or pred["category"] != gt["category"]:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= 0.5:
                matched_gt.add(best_idx)
                class_stats[pred["category"]]["matched"] += 1

    mean_recall = float(np.mean([r["recall"] for r in per_image]))
    mean_precision = float(np.mean([r["precision"] for r in per_image]))
    mean_iou = float(np.mean([r["mean_matched_iou"] for r in per_image if r["num_matched"] > 0] or [0]))

    per_class_recall = {
        cat: (v["matched"] / v["gt"] if v["gt"] else 0.0) for cat, v in class_stats.items()
    }

    summary = {
        "split": split,
        "num_images": len(per_image),
        "seed": seed,
        "model": model_name,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "mean_matched_iou": mean_iou,
        "per_class_recall": per_class_recall,
        "per_image": [
            {k: v for k, v in r.items() if k not in ("pred_boxes", "gt_boxes")} for r in per_image
        ],
    }

    out_json = METRICS_DIR / f"detection_baseline_{split}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_detection_preview_grid(image_paths[: min(3, len(image_paths))], label_index, model, split)
    per_class_path = FIGURES_DIR / f"detection_per_class_recall_{split}.png"
    save_per_class_recall_chart(
        per_class_recall,
        output_path=per_class_path,
        title=f"YOLOv8 per-class recall @ IoU 0.5 — {split} (N={len(per_image)})",
    )
    print(f"Mean recall: {mean_recall:.3f}  precision: {mean_precision:.3f}  matched IoU: {mean_iou:.3f}")
    print(f"Saved metrics: {out_json}")
    print(f"Saved per-image GT|Pred panels: {vis_dir}")
    return summary


def save_detection_preview_grid(
    image_paths: list[Path],
    label_index: dict,
    model: YOLO,
    split: str,
) -> None:
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        gt_boxes = load_detection_boxes(resolve_label_path(image_path.stem, label_index))
        pred_boxes = yolo_predictions_to_boxes(model.predict(image, verbose=False)[0])
        metrics = evaluate_detection_boxes(gt_boxes, pred_boxes)

        panels = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(draw_detection_boxes(image, gt_boxes), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(draw_detection_boxes(image, pred_boxes), cv2.COLOR_BGR2RGB),
        ]
        titles = ["Original", f"GT ({len(gt_boxes)} boxes)", f"YOLO ({len(pred_boxes)} boxes)"]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(
                    f"{image_path.stem}\nR={metrics['recall']:.2f}",
                    rotation=90,
                    labelpad=36,
                    fontsize=8,
                )

    fig.suptitle(f"Detection baseline — {split} (YOLOv8)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / f"detection_baseline_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview grid: {out}")


def _eval_image_detection(model: YOLO, image: np.ndarray, gt_boxes: list[dict]) -> dict:
    pred_boxes = yolo_predictions_to_boxes(model.predict(image, verbose=False)[0])
    return evaluate_detection_boxes(gt_boxes, pred_boxes)


def run_robustness(split: str, num_images: int, seed: int, model_name: str) -> dict:
    """Evaluate recall on distorted/enhanced images and plot vs intensity."""
    ensure_output_dirs()
    label_index = build_label_index()
    image_paths = select_paired_images(split, num_images, seed, label_index)
    if not image_paths:
        raise SystemExit(f"No images with detection + seg GT for split '{split}'.")

    baseline_path = METRICS_DIR / f"detection_baseline_{split}.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"Loaded baseline recall: {baseline['mean_recall']:.3f}")
    else:
        baseline = run_baseline(split, num_images, seed, model_name)

    model = YOLO(model_name)
    distorted: dict = {}
    enhanced: dict = {}

    for distortion in DISTORTION_TYPES:
        distorted[distortion] = {}
        enhanced[distortion] = {}
        for level in default_levels(distortion):
            level_key = str(level)
            rng = np.random.default_rng(level_seed(seed, distortion, level))
            recalls_d, recalls_e = [], []
            stats_d = new_detection_class_stats()
            stats_e = new_detection_class_stats()

            for image_path in tqdm(image_paths, desc=f"det {distortion}/{level_key}"):
                clean = cv2.imread(str(image_path))
                if clean is None:
                    continue
                gt_boxes = load_detection_boxes(resolve_label_path(image_path.stem, label_index))
                dist_img = apply_distortion(clean, distortion, level, rng=rng)
                enh_img = enhance_for_distortion(dist_img, distortion)
                pred_d = yolo_predictions_to_boxes(model.predict(dist_img, verbose=False)[0])
                pred_e = yolo_predictions_to_boxes(model.predict(enh_img, verbose=False)[0])
                recalls_d.append(evaluate_detection_boxes(gt_boxes, pred_d)["recall"])
                recalls_e.append(evaluate_detection_boxes(gt_boxes, pred_e)["recall"])
                update_detection_class_stats(stats_d, gt_boxes, pred_d)
                update_detection_class_stats(stats_e, gt_boxes, pred_e)

            distorted[distortion][level_key] = {
                "level": level,
                "mean_recall": float(np.mean(recalls_d)),
                "per_class_recall": per_class_recall_from_stats(stats_d),
            }
            enhanced[distortion][level_key] = {
                "level": level,
                "mean_recall": float(np.mean(recalls_e)),
                "per_class_recall": per_class_recall_from_stats(stats_e),
            }
            print(
                f"  {distortion} {level_key}: "
                f"dist={distorted[distortion][level_key]['mean_recall']:.3f}  "
                f"enh={enhanced[distortion][level_key]['mean_recall']:.3f}"
            )

    combined = {
        "split": split,
        "num_images": len(image_paths),
        "seed": seed,
        "model": model_name,
        "baseline": {"mean_recall": baseline["mean_recall"]},
        "distorted": distorted,
        "enhanced": enhanced,
    }
    out_json = METRICS_DIR / f"detection_robustness_{split}.json"
    if out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        for key in ("finetuned", "finetuned_eval_note"):
            if key in prev:
                combined[key] = prev[key]
    out_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    plot_path = FIGURES_DIR / f"detection_robustness_{split}.png"
    save_robustness_summary_plot(
        combined,
        plot_path,
        metric_key="mean_recall",
        ylabel="Recall @ IoU 0.5",
        title="YOLOv8 detection recall vs distortion intensity",
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {plot_path}")
    return combined


def run_distorted(split: str, num_images: int | None, model_name: str) -> None:
    run_robustness(split, num_images or 10, 42, model_name)


def run_enhanced(split: str, num_images: int | None, model_name: str) -> None:
    run_robustness(split, num_images or 10, 42, model_name)


def run_finetune(
    split: str,
    distortion: str,
    model_name: str,
    *,
    level: float | int | None = None,
    num_train: int = 300,
    num_val: int = 50,
    seed: int = 42,
    epochs: int = 30,
    batch: int = 8,
    imgsz: int = 640,
    rebuild_dataset: bool = False,
) -> dict:
    """Build distorted YOLO dataset and fine-tune YOLOv8."""
    ensure_output_dirs()
    level = level if level is not None else default_finetune_level(distortion)
    if distortion == "jpeg":
        level = int(level)

    build_distorted_yolo_dataset(
        distortion,
        level,
        split=split,
        num_train=num_train,
        num_val=num_val,
        seed=seed,
        rebuild=rebuild_dataset,
    )
    root = dataset_root(distortion, level)
    yaml_path = root / "dataset.yaml"
    run_name = f"yolo_{distortion}_{level_tag(distortion, level)}"

    model = YOLO(model_name)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(FINETUNE_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    best_weights = FINETUNE_DIR / run_name / "weights" / "best.pt"
    summary = {
        "split": split,
        "distortion": distortion,
        "level": level,
        "seed": seed,
        "num_train": num_train,
        "num_val": num_val,
        "epochs": epochs,
        "base_model": model_name,
        "dataset_yaml": str(yaml_path),
        "best_weights": str(best_weights),
        "run_name": run_name,
    }
    out_json = METRICS_DIR / f"detection_finetune_{distortion}_{level_tag(distortion, level)}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Fine-tuning complete. Best weights: {best_weights}")
    print(f"Saved training summary: {out_json}")
    return summary


def _load_gt_boxes(stem: str, image: np.ndarray, root: Path, label_index: dict) -> list[dict]:
    """Load GT boxes from BDD JSON when available, else from YOLO labels in the dataset."""
    label_path = resolve_label_path(stem, label_index)
    gt_boxes = load_detection_boxes(label_path)
    if gt_boxes:
        return gt_boxes
    yolo_label = root / "labels" / "val" / f"{stem}.txt"
    h, w = image.shape[:2]
    return load_yolo_label_boxes(yolo_label, w, h)


def save_finetune_preview_grid(
    stems: list[str],
    root: Path,
    pretrained: YOLO,
    finetuned: YOLO,
    label_index: dict,
    *,
    distortion: str,
    tag: str,
    conf_thresh: float = 0.25,
) -> Path:
    """Side-by-side pretrained vs fine-tuned detections on distorted val images."""
    n = len(stems)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Pretrained YOLO", "Fine-tuned YOLO"]

    for row, stem in enumerate(stems):
        img_path = root / "images" / "val" / f"{stem}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        gt_boxes = _load_gt_boxes(stem, image, root, label_index)
        pred_pre = yolo_predictions_to_boxes(
            pretrained.predict(image, verbose=False)[0], conf_thresh=conf_thresh
        )
        pred_ft = yolo_predictions_to_boxes(
            finetuned.predict(image, verbose=False)[0], conf_thresh=conf_thresh
        )
        m_pre = evaluate_detection_boxes(gt_boxes, pred_pre)
        m_ft = evaluate_detection_boxes(gt_boxes, pred_ft)

        panels = [
            cv2.cvtColor(draw_detection_boxes(image, pred_pre), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(draw_detection_boxes(image, pred_ft), cv2.COLOR_BGR2RGB),
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(
                    f"{stem}\nR: {m_pre['recall']:.2f} → {m_ft['recall']:.2f}",
                    rotation=90,
                    labelpad=40,
                    fontsize=8,
                )

    fig.suptitle(f"Fine-tuning detection preview — {distortion} ({tag})", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / f"detection_finetune_preview_{distortion}_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview grid: {out}")
    return out


def run_finetune_plots(
    distortion: str,
    level: float | int | None = None,
    *,
    num_preview: int = 3,
    pretrained_model: str = "yolov8n.pt",
) -> None:
    """Generate training curves and qualitative preview figures for a fine-tune run."""
    ensure_output_dirs()
    level = level if level is not None else default_finetune_level(distortion)
    if distortion == "jpeg":
        level = int(level)

    tag = level_tag(distortion, level)
    run_name = f"yolo_{distortion}_{tag}"
    run_dir = FINETUNE_DIR / run_name
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise SystemExit(f"Training results not found: {results_csv}")

    curves_path = FIGURES_DIR / f"detection_finetune_training_{distortion}_{tag}.png"
    save_finetune_training_curves(
        results_csv,
        curves_path,
        title=f"YOLOv8 fine-tuning — {distortion} ({tag})",
    )
    print(f"Saved training curves: {curves_path}")

    ultralytics_plot = run_dir / "results.png"
    if ultralytics_plot.exists():
        import shutil

        dest = FIGURES_DIR / f"detection_finetune_ultralytics_{distortion}_{tag}.png"
        shutil.copy2(ultralytics_plot, dest)
        print(f"Copied Ultralytics summary: {dest}")

    manifest = load_manifest(distortion, level)
    root = dataset_root(distortion, level)
    finetuned_weights = run_dir / "weights" / "best.pt"
    if not finetuned_weights.exists():
        print("Skipping preview grid — fine-tuned weights not found.")
        return

    eval_json = METRICS_DIR / f"detection_finetune_eval_{distortion}_{tag}.json"
    val_stems = manifest["val"]
    if eval_json.exists():
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
        ranked = sorted(
            eval_data.get("per_image", []),
            key=lambda row: row["finetuned_recall"] - row["pretrained_recall"],
            reverse=True,
        )
        preview_stems = [row["image"].replace(".jpg", "") for row in ranked[:num_preview]]
    else:
        preview_stems = val_stems[:num_preview]

    if not preview_stems:
        return

    label_index = build_label_index()
    save_finetune_preview_grid(
        preview_stems,
        root,
        YOLO(pretrained_model),
        YOLO(str(finetuned_weights)),
        label_index,
        distortion=distortion,
        tag=tag,
    )


def run_finetune_eval(
    split: str,
    distortion: str,
    pretrained_model: str,
    *,
    level: float | int | None = None,
    seed: int = 42,
    conf_thresh: float = 0.25,
) -> dict:
    """Compare pretrained vs fine-tuned recall on distorted val images."""
    ensure_output_dirs()
    level = level if level is not None else default_finetune_level(distortion)
    if distortion == "jpeg":
        level = int(level)

    manifest = load_manifest(distortion, level)
    run_name = f"yolo_{distortion}_{level_tag(distortion, level)}"
    finetuned_weights = FINETUNE_DIR / run_name / "weights" / "best.pt"
    if not finetuned_weights.exists():
        raise SystemExit(
            f"Fine-tuned weights not found: {finetuned_weights}\n"
            "Run --mode finetune first."
        )

    label_index = build_label_index()
    root = dataset_root(distortion, level)
    val_stems = manifest["val"]
    if not val_stems:
        raise SystemExit("No val images in dataset manifest.")

    pretrained = YOLO(pretrained_model)
    finetuned = YOLO(str(finetuned_weights))

    pretrained_recalls: list[float] = []
    finetuned_recalls: list[float] = []
    enhanced_recalls: list[float] = []
    per_image: list[dict] = []

    for stem in tqdm(val_stems, desc="finetune eval"):
        val_img_path = root / "images" / "val" / f"{stem}.jpg"
        if not val_img_path.exists():
            continue
        distorted = cv2.imread(str(val_img_path))
        if distorted is None:
            continue

        gt_boxes = _load_gt_boxes(stem, distorted, root, label_index)
        if not gt_boxes:
            continue

        enhanced = enhance_for_distortion(distorted, distortion)
        pred_pre = yolo_predictions_to_boxes(
            pretrained.predict(distorted, verbose=False)[0], conf_thresh=conf_thresh
        )
        pred_enh = yolo_predictions_to_boxes(
            pretrained.predict(enhanced, verbose=False)[0], conf_thresh=conf_thresh
        )
        pred_ft = yolo_predictions_to_boxes(
            finetuned.predict(distorted, verbose=False)[0], conf_thresh=conf_thresh
        )
        m_pre = evaluate_detection_boxes(gt_boxes, pred_pre)
        m_enh = evaluate_detection_boxes(gt_boxes, pred_enh)
        m_ft = evaluate_detection_boxes(gt_boxes, pred_ft)
        pretrained_recalls.append(m_pre["recall"])
        enhanced_recalls.append(m_enh["recall"])
        finetuned_recalls.append(m_ft["recall"])
        per_image.append(
            {
                "image": f"{stem}.jpg",
                "pretrained_recall": m_pre["recall"],
                "enhanced_recall": m_enh["recall"],
                "finetuned_recall": m_ft["recall"],
            }
        )

    if not per_image:
        raise SystemExit(
            "No val images evaluated. Check that data/yolo_distorted/ contains "
            "images and labels for the requested distortion."
        )

    mean_pretrained = float(np.mean(pretrained_recalls))
    mean_enhanced = float(np.mean(enhanced_recalls))
    mean_finetuned = float(np.mean(finetuned_recalls))
    tag = level_tag(distortion, level)
    summary = {
        "split": split,
        "distortion": distortion,
        "level": level,
        "seed": seed,
        "num_val_images": len(per_image),
        "pretrained_model": pretrained_model,
        "finetuned_weights": str(finetuned_weights),
        "mean_recall_pretrained": mean_pretrained,
        "mean_recall_enhanced": mean_enhanced,
        "mean_recall_finetuned": mean_finetuned,
        "per_image": per_image,
    }
    out_json = METRICS_DIR / f"detection_finetune_eval_{distortion}_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = FIGURES_DIR / f"detection_finetune_{distortion}_{tag}.png"
    baseline_recall = load_detection_clean_baseline_recall(split)
    save_comparison_bars(
        ["Pretrained", enhancement_label(distortion), "Fine-tuned"],
        [mean_pretrained, mean_enhanced, mean_finetuned],
        ylabel="Recall @ IoU 0.5",
        title=f"Detection on distorted val — {distortion} ({tag})",
        output_path=plot_path,
        colors=["#95a5a6", "#f39c12", "#2ecc71"],
        baseline=baseline_recall,
    )
    print(
        f"Val recall — pretrained: {mean_pretrained:.3f}  "
        f"enhanced: {mean_enhanced:.3f}  "
        f"fine-tuned: {mean_finetuned:.3f}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {plot_path}")

    run_finetune_plots(distortion, level, pretrained_model=pretrained_model)
    return summary


def _distortion_display_name(distortion: str) -> str:
    return {
        "noise": "Noise",
        "low_light": "Low Light",
        "jpeg": "JPEG",
    }.get(distortion, distortion.replace("_", " ").title())


def _finetune_condition_label(distortion: str, level: float | int) -> str:
    if distortion == "noise":
        return f"Noise (SNR {level:g} dB)"
    if distortion == "low_light":
        return f"Low light (γ={level:g})"
    if distortion == "jpeg":
        return f"JPEG (Q={int(level)})"
    return f"{distortion} ({level})"


def _best_finetune_preview_stem(distortion: str, level: float | int) -> str | None:
    tag = level_tag(distortion, level)
    eval_json = METRICS_DIR / f"detection_finetune_eval_{distortion}_{tag}.json"
    if eval_json.exists():
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
        ranked = sorted(
            eval_data.get("per_image", []),
            key=lambda row: row["finetuned_recall"] - row["pretrained_recall"],
            reverse=True,
        )
        if ranked:
            return ranked[0]["image"].replace(".jpg", "")

    manifest = load_manifest(distortion, level)
    val_stems = manifest.get("val", [])
    return val_stems[0] if val_stems else None


def save_finetune_summary_preview(
    *,
    pretrained_model: str = "yolov8n.pt",
    examples: list[tuple[str, float | int]] | None = None,
    conf_thresh: float = 0.25,
) -> Path:
    """Qualitative fine-tuning examples — one row per distortion type."""
    ensure_output_dirs()
    if examples is None:
        examples = [(distortion, default_finetune_level(distortion)) for distortion in DISTORTION_TYPES]

    label_index = build_label_index()
    pretrained = YOLO(pretrained_model)
    rows: list[dict] = []

    for distortion, level in examples:
        if distortion == "jpeg":
            level = int(level)
        tag = level_tag(distortion, level)
        root = dataset_root(distortion, level)
        weights = FINETUNE_DIR / f"yolo_{distortion}_{tag}" / "weights" / "best.pt"
        stem = _best_finetune_preview_stem(distortion, level)
        if stem is None or not weights.exists():
            continue

        img_path = root / "images" / "val" / f"{stem}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        finetuned = YOLO(str(weights))
        gt_boxes = _load_gt_boxes(stem, image, root, label_index)
        pred_pre = yolo_predictions_to_boxes(
            pretrained.predict(image, verbose=False)[0], conf_thresh=conf_thresh
        )
        pred_ft = yolo_predictions_to_boxes(
            finetuned.predict(image, verbose=False)[0], conf_thresh=conf_thresh
        )
        m_pre = evaluate_detection_boxes(gt_boxes, pred_pre)
        m_ft = evaluate_detection_boxes(gt_boxes, pred_ft)
        rows.append(
            {
                "distortion": distortion,
                "distortion_label": _distortion_display_name(distortion),
                "label": _finetune_condition_label(distortion, level),
                "stem": stem,
                "image": image,
                "gt_boxes": gt_boxes,
                "pred_pre": pred_pre,
                "pred_ft": pred_ft,
                "recall_pre": m_pre["recall"],
                "recall_ft": m_ft["recall"],
            }
        )

    if not rows:
        raise SystemExit("No fine-tuned weights / val images found for summary preview.")

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Pretrained YOLO", "Fine-tuned YOLO"]
    for row_idx, row in enumerate(rows):
        panels = [
            cv2.cvtColor(draw_detection_boxes(row["image"], row["pred_pre"]), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(draw_detection_boxes(row["image"], row["pred_ft"]), cv2.COLOR_BGR2RGB),
        ]
        for col, panel in enumerate(panels):
            axes[row_idx, col].imshow(panel)
            axes[row_idx, col].axis("off")
            if row_idx == 0:
                axes[row_idx, col].set_title(col_titles[col], fontsize=11)
            if col == 0:
                axes[row_idx, col].text(
                    0.02,
                    0.98,
                    row["distortion_label"],
                    transform=axes[row_idx, col].transAxes,
                    fontsize=13,
                    fontweight="bold",
                    va="top",
                    ha="left",
                    color="white",
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "black", "alpha": 0.7},
                )
                axes[row_idx, col].text(
                    0.02,
                    0.88,
                    row["label"],
                    transform=axes[row_idx, col].transAxes,
                    fontsize=9,
                    va="top",
                    ha="left",
                    color="white",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#333333", "alpha": 0.65},
                )
                axes[row_idx, col].set_ylabel(
                    f"{row['stem']}\nR: {row['recall_pre']:.2f} → {row['recall_ft']:.2f}",
                    rotation=90,
                    labelpad=48,
                    fontsize=8,
                )

    fig.suptitle("YOLO fine-tuning — qualitative examples (largest recall gain per type)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "detection_finetune_summary_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary preview: {out}")
    return out


def run_finetune_summary(summary_path: Path | None = None, pretrained_model: str = "yolov8n.pt") -> dict:
    """Generate summary figures from YOLO fine-tune batch results."""
    ensure_output_dirs()
    grouped, batch = load_detection_finetune_batch_results(summary_path)

    if not any(grouped.values()):
        raise SystemExit("No fine-tune batch results found in summary JSON.")

    subtitle = (
        f"{batch.get('num_train', 500)} train / {batch.get('num_val', 100)} val · "
        f"{batch.get('epochs', '?')} epochs"
    )
    baseline_recall = load_detection_clean_baseline_recall("train")
    recall_path = FIGURES_DIR / "detection_finetune_summary_recall.png"
    gain_path = FIGURES_DIR / "detection_finetune_summary_gain.png"
    table_path = FIGURES_DIR / "detection_finetune_summary_table.png"

    save_detection_finetune_summary_recall_plot(
        grouped,
        recall_path,
        subtitle=subtitle,
        baseline_recall=baseline_recall,
    )
    save_detection_finetune_summary_gain_plot(grouped, gain_path)
    save_detection_finetune_summary_table_plot(
        grouped, table_path, baseline_recall=baseline_recall
    )

    try:
        save_finetune_summary_preview(pretrained_model=pretrained_model)
    except SystemExit as exc:
        print(exc)

    print(f"Saved summary recall plot: {recall_path}")
    print(f"Saved summary gain plot: {gain_path}")
    print(f"Saved summary table plot: {table_path}")
    return {"grouped": grouped, "batch": batch}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 object detection evaluation.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=("baseline", "robustness", "distorted", "enhanced", "finetune", "finetune-eval", "finetune-plots", "finetune-summary", "all"),
    )
    parser.add_argument("--distortion", default="noise", choices=("noise", "low_light", "jpeg"))
    parser.add_argument(
        "--level",
        type=float,
        default=None,
        help="Distortion intensity for finetune (default: mid level per type)",
    )
    parser.add_argument("--num-train", type=int, default=300, help="Training images for YOLO dataset")
    parser.add_argument("--num-val", type=int, default=50, help="Validation images for YOLO dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument(
        "--num-preview",
        type=int,
        default=3,
        help="Images per job for finetune-plots preview grids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    if args.mode == "baseline":
        run_baseline(args.split, args.num_images, args.seed, args.model)
    elif args.mode == "robustness":
        run_robustness(args.split, args.num_images, args.seed, args.model)
    elif args.mode == "distorted":
        run_distorted(args.split, args.num_images, args.model)
    elif args.mode == "enhanced":
        run_enhanced(args.split, args.num_images, args.model)
    elif args.mode == "finetune":
        run_finetune(
            args.split,
            args.distortion,
            args.model,
            level=args.level,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
            epochs=args.epochs,
            batch=args.batch,
            rebuild_dataset=args.rebuild_dataset,
        )
    elif args.mode == "finetune-eval":
        run_finetune_eval(
            args.split,
            args.distortion,
            args.model,
            level=args.level,
            seed=args.seed,
        )
    elif args.mode == "finetune-plots":
        run_finetune_plots(args.distortion, args.level, pretrained_model=args.model)
    elif args.mode == "finetune-summary":
        run_finetune_summary(pretrained_model=args.model)
    elif args.mode == "all":
        run_baseline(args.split, args.num_images, args.seed, args.model)
        run_robustness(args.split, args.num_images, args.seed, args.model)


if __name__ == "__main__":
    main()
