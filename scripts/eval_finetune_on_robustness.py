"""Evaluate fine-tuned YOLO and SegFormer on the same 100-image robustness set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bdd100k_utils import (
    SEG_CLASS_NAMES,
    build_label_index,
    load_detection_boxes,
    load_seg_mask,
    resolve_label_path,
    select_paired_images,
)
from src.distortions import apply_distortion, default_levels, level_tag
from src.evaluate import (
    evaluate_detection_boxes,
    new_detection_class_stats,
    new_segmentation_class_accumulator,
    per_class_miou_from_accumulator,
    per_class_recall_from_stats,
    save_robustness_summary_plot,
    update_detection_class_stats,
    update_segmentation_class_accumulator,
)
from src.paths import DISTORTION_TYPES, FIGURES_DIR, FINETUNE_DIR, METRICS_DIR, ensure_output_dirs
from src.robustness import level_seed
from src.run_detection import yolo_predictions_to_boxes
from src.run_segmentation import _eval_image_segmentation, _load_segformer_model, _segformer_run_name


def _merge_finetuned_into_robustness(
    robustness_path: Path,
    finetuned: dict,
    *,
    metric_key: str,
    plot_path: Path,
    ylabel: str,
    title: str,
) -> dict:
    if robustness_path.exists():
        combined = json.loads(robustness_path.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"Robustness metrics not found: {robustness_path}")

    combined["finetuned"] = finetuned
    combined["finetuned_eval_note"] = (
        "Fine-tuned checkpoints evaluated on the same images as distorted/enhanced "
        "(robustness set), using identical distortion RNG seeds."
    )
    robustness_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    save_robustness_summary_plot(
        combined,
        plot_path,
        metric_key=metric_key,
        ylabel=ylabel,
        title=title,
    )
    return combined


def eval_detection_finetune_robustness(
    split: str = "train",
    num_images: int = 100,
    seed: int = 42,
    pretrained_model: str = "yolov8n.pt",
) -> dict:
    ensure_output_dirs()
    label_index = build_label_index()
    image_paths = select_paired_images(split, num_images, seed, label_index)
    if not image_paths:
        raise SystemExit(f"No images for split '{split}'.")

    finetuned: dict = {d: {} for d in DISTORTION_TYPES}

    for distortion in DISTORTION_TYPES:
        for level in default_levels(distortion):
            level_key = str(level)
            tag = level_tag(distortion, level)
            weights = FINETUNE_DIR / f"yolo_{distortion}_{tag}" / "weights" / "best.pt"
            if not weights.exists():
                print(f"Skipping {distortion}/{level_key}: {weights} missing")
                continue

            ft_model = YOLO(str(weights))
            rng = np.random.default_rng(level_seed(seed, distortion, level))
            recalls: list[float] = []
            stats = new_detection_class_stats()

            for image_path in tqdm(image_paths, desc=f"yolo FT robust {distortion}/{level_key}"):
                clean = cv2.imread(str(image_path))
                if clean is None:
                    continue
                gt_boxes = load_detection_boxes(resolve_label_path(image_path.stem, label_index))
                dist_img = apply_distortion(clean, distortion, level, rng=rng)
                pred = yolo_predictions_to_boxes(ft_model.predict(dist_img, verbose=False)[0])
                recalls.append(evaluate_detection_boxes(gt_boxes, pred)["recall"])
                update_detection_class_stats(stats, gt_boxes, pred)

            finetuned[distortion][level_key] = {
                "level": level,
                "mean_recall": float(np.mean(recalls)),
                "per_class_recall": per_class_recall_from_stats(stats),
            }
            print(f"  {distortion} {level_key}: FT recall={finetuned[distortion][level_key]['mean_recall']:.3f}")

    out_path = METRICS_DIR / f"detection_robustness_{split}.json"
    combined = _merge_finetuned_into_robustness(
        out_path,
        finetuned,
        metric_key="mean_recall",
        plot_path=FIGURES_DIR / f"detection_robustness_{split}.png",
        ylabel="Recall @ IoU 0.5",
        title="YOLOv8 detection recall vs distortion intensity",
    )
    print(f"Updated {out_path}")
    return combined


def eval_segmentation_finetune_robustness(
    split: str = "train",
    num_images: int = 100,
    seed: int = 42,
    model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
) -> dict:
    ensure_output_dirs()
    label_index = build_label_index()
    image_paths = select_paired_images(split, num_images, seed, label_index)
    if not image_paths:
        raise SystemExit(f"No images for split '{split}'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetuned: dict = {d: {} for d in DISTORTION_TYPES}

    for distortion in DISTORTION_TYPES:
        for level in default_levels(distortion):
            level_key = str(level)
            run_name = _segformer_run_name(distortion, level)
            ft_dir = FINETUNE_DIR / run_name
            if not ft_dir.exists():
                print(f"Skipping {distortion}/{level_key}: {ft_dir} missing")
                continue

            ft_model, ft_processor = _load_segformer_model(str(ft_dir), device)
            rng = np.random.default_rng(level_seed(seed, distortion, level))
            mious: list[float] = []
            acc = new_segmentation_class_accumulator()

            for image_path in tqdm(image_paths, desc=f"seg FT robust {distortion}/{level_key}"):
                clean = cv2.imread(str(image_path))
                gt_mask = load_seg_mask(split, image_path.stem)
                if clean is None or gt_mask is None:
                    continue
                dist_img = apply_distortion(clean, distortion, level, rng=rng)
                res = _eval_image_segmentation(ft_model, ft_processor, device, dist_img, gt_mask)
                mious.append(res["miou"])
                update_segmentation_class_accumulator(acc, res["per_class_iou"])

            finetuned[distortion][level_key] = {
                "level": level,
                "mean_miou": float(np.mean(mious)),
                "per_class_miou": per_class_miou_from_accumulator(acc, SEG_CLASS_NAMES),
            }
            print(f"  {distortion} {level_key}: FT mIoU={finetuned[distortion][level_key]['mean_miou']:.3f}")

    out_path = METRICS_DIR / f"segmentation_robustness_{split}.json"
    combined = _merge_finetuned_into_robustness(
        out_path,
        finetuned,
        metric_key="mean_miou",
        plot_path=FIGURES_DIR / f"segmentation_robustness_{split}.png",
        ylabel="mIoU",
        title="SegFormer mIoU vs distortion intensity",
    )
    print(f"Updated {out_path}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuned model eval on robustness set.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        default="all",
        choices=("detection", "segmentation", "all"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task in ("detection", "all"):
        eval_detection_finetune_robustness(args.split, args.num_images, args.seed)
    if args.task in ("segmentation", "all"):
        eval_segmentation_finetune_robustness(args.split, args.num_images, args.seed)


if __name__ == "__main__":
    main()
