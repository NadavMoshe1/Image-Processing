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
from src.enhancements import enhance_for_distortion
from src.evaluate import (
    box_iou,
    evaluate_detection_boxes,
    save_comparison_bars,
    save_per_class_recall_chart,
    save_robustness_summary_plot,
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

            for image_path in tqdm(image_paths, desc=f"det {distortion}/{level_key}"):
                clean = cv2.imread(str(image_path))
                if clean is None:
                    continue
                gt_boxes = load_detection_boxes(resolve_label_path(image_path.stem, label_index))
                dist_img = apply_distortion(clean, distortion, level, rng=rng)
                enh_img = enhance_for_distortion(dist_img, distortion)
                recalls_d.append(_eval_image_detection(model, dist_img, gt_boxes)["recall"])
                recalls_e.append(_eval_image_detection(model, enh_img, gt_boxes)["recall"])

            distorted[distortion][level_key] = {
                "level": level,
                "mean_recall": float(np.mean(recalls_d)),
            }
            enhanced[distortion][level_key] = {
                "level": level,
                "mean_recall": float(np.mean(recalls_e)),
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
    per_image: list[dict] = []

    for stem in tqdm(val_stems, desc="finetune eval"):
        val_img_path = root / "images" / "val" / f"{stem}.jpg"
        if not val_img_path.exists():
            continue
        distorted = cv2.imread(str(val_img_path))
        if distorted is None:
            continue

        label_path = resolve_label_path(stem, label_index)
        gt_boxes = load_detection_boxes(label_path)
        if not gt_boxes:
            continue

        pred_pre = yolo_predictions_to_boxes(
            pretrained.predict(distorted, verbose=False)[0], conf_thresh=conf_thresh
        )
        pred_ft = yolo_predictions_to_boxes(
            finetuned.predict(distorted, verbose=False)[0], conf_thresh=conf_thresh
        )
        m_pre = evaluate_detection_boxes(gt_boxes, pred_pre)
        m_ft = evaluate_detection_boxes(gt_boxes, pred_ft)
        pretrained_recalls.append(m_pre["recall"])
        finetuned_recalls.append(m_ft["recall"])
        per_image.append(
            {
                "image": f"{stem}.jpg",
                "pretrained_recall": m_pre["recall"],
                "finetuned_recall": m_ft["recall"],
            }
        )

    mean_pretrained = float(np.mean(pretrained_recalls))
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
        "mean_recall_finetuned": mean_finetuned,
        "per_image": per_image,
    }
    out_json = METRICS_DIR / f"detection_finetune_eval_{distortion}_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = FIGURES_DIR / f"detection_finetune_{distortion}_{tag}.png"
    save_comparison_bars(
        ["Pretrained", "Fine-tuned"],
        [mean_pretrained, mean_finetuned],
        ylabel="Recall @ IoU 0.5",
        title=f"Detection on distorted val — {distortion} ({tag})",
        output_path=plot_path,
    )
    print(
        f"Val recall — pretrained: {mean_pretrained:.3f}  "
        f"fine-tuned: {mean_finetuned:.3f}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {plot_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 object detection evaluation.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=("baseline", "robustness", "distorted", "enhanced", "finetune", "finetune-eval", "all"),
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
    elif args.mode == "all":
        run_baseline(args.split, args.num_images, args.seed, args.model)
        run_robustness(args.split, args.num_images, args.seed, args.model)


if __name__ == "__main__":
    main()
