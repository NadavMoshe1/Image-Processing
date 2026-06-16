"""SegFormer semantic segmentation baseline and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from src.bdd100k_utils import (
    SEG_CLASS_NAMES,
    build_label_index,
    load_seg_color,
    load_seg_mask,
    overlay_seg_mask,
    select_paired_images,
)
from src.distortions import apply_distortion, default_levels
from src.enhancements import enhance_for_distortion
from src.evaluate import compute_miou, save_per_class_miou_chart, save_robustness_summary_plot
from src.paths import BASELINE_SEGMENTATION_DIR, DISTORTION_TYPES, FIGURES_DIR, METRICS_DIR, ensure_output_dirs
from src.robustness import level_seed

# Cityscapes palette (19 classes) for prediction overlay — BGR
CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in np.unique(mask):
        if cls_id == 255:
            continue
        if 0 <= cls_id < len(CITYSCAPES_COLORS):
            color[mask == cls_id] = CITYSCAPES_COLORS[cls_id]
    return color


def predict_segmentation(
    model,
    processor,
    image: np.ndarray,
    device: str,
) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(image.shape[0], image.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def run_baseline(split: str, num_images: int, seed: int, model_id: str) -> dict:
    ensure_output_dirs()
    label_index = build_label_index()
    image_paths = select_paired_images(split, num_images, seed, label_index)
    if not image_paths:
        raise SystemExit(f"No images with detection + seg GT for split '{split}'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SegFormer on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    per_image = []
    class_ious: dict[int, list[float]] = {}

    vis_dir = BASELINE_SEGMENTATION_DIR / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="SegFormer"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        gt_mask = load_seg_mask(split, image_path.stem)
        gt_color = load_seg_color(split, image_path.stem)
        if gt_mask is None:
            continue

        pred_mask = predict_segmentation(model, processor, image, device)
        metrics = compute_miou(pred_mask, gt_mask)

        per_image.append(
            {
                "image": image_path.name,
                "miou": metrics["miou"],
                "per_class_iou": {str(k): v for k, v in metrics["per_class_iou"].items()},
            }
        )

        for cls_id, iou in metrics["per_class_iou"].items():
            class_ious.setdefault(cls_id, []).append(iou)

        pred_color = mask_to_color(pred_mask)
        gt_overlay = overlay_seg_mask(image, gt_color) if gt_color is not None else image
        pred_overlay = overlay_seg_mask(image, pred_color)
        panel = np.hstack([image, gt_overlay, pred_overlay])
        cv2.imwrite(str(vis_dir / f"{image_path.stem}_gt_pred.jpg"), panel)

    mean_miou = float(np.mean([r["miou"] for r in per_image]))
    per_class_mean = {
        SEG_CLASS_NAMES[cls_id] if cls_id < len(SEG_CLASS_NAMES) else str(cls_id): float(np.mean(vals))
        for cls_id, vals in class_ious.items()
    }

    summary = {
        "split": split,
        "num_images": len(per_image),
        "seed": seed,
        "model": model_id,
        "mean_miou": mean_miou,
        "per_class_miou": per_class_mean,
        "per_image": per_image,
    }

    out_json = METRICS_DIR / f"segmentation_baseline_{split}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_segmentation_preview_grid(image_paths[: min(3, len(image_paths))], model, processor, device, split)
    save_per_class_miou_chart(
        per_class_mean,
        output_path=FIGURES_DIR / f"segmentation_per_class_miou_{split}.png",
        title=f"SegFormer per-class IoU — {split} (N={len(per_image)})",
        mean_miou=mean_miou,
    )
    print(f"Mean mIoU: {mean_miou:.3f}")
    print(f"Saved metrics: {out_json}")
    print(f"Saved per-image GT|Pred panels: {vis_dir}")
    return summary


def save_segmentation_preview_grid(
    image_paths: list[Path],
    model,
    processor,
    device: str,
    split: str,
) -> None:
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        gt_color = load_seg_color(split, image_path.stem)
        gt_mask = load_seg_mask(split, image_path.stem)
        pred_mask = predict_segmentation(model, processor, image, device)
        miou = compute_miou(pred_mask, gt_mask)["miou"] if gt_mask is not None else 0.0

        panels = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(
                overlay_seg_mask(image, gt_color) if gt_color is not None else image,
                cv2.COLOR_BGR2RGB,
            ),
            cv2.cvtColor(overlay_seg_mask(image, mask_to_color(pred_mask)), cv2.COLOR_BGR2RGB),
        ]
        titles = ["Original", "GT segmentation", "SegFormer prediction"]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(
                    f"{image_path.stem}\nmIoU={miou:.2f}",
                    rotation=90,
                    labelpad=36,
                    fontsize=8,
                )

    fig.suptitle(f"Segmentation baseline — {split} (SegFormer)", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / f"segmentation_baseline_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview grid: {out}")


def _eval_image_segmentation(
    model,
    processor,
    device: str,
    image: np.ndarray,
    gt_mask: np.ndarray,
) -> dict:
    pred_mask = predict_segmentation(model, processor, image, device)
    return compute_miou(pred_mask, gt_mask)


def run_robustness(split: str, num_images: int, seed: int, model_id: str) -> dict:
    """Evaluate mIoU on distorted/enhanced images and plot vs intensity."""
    ensure_output_dirs()
    label_index = build_label_index()
    image_paths = select_paired_images(split, num_images, seed, label_index)
    if not image_paths:
        raise SystemExit(f"No images with detection + seg GT for split '{split}'.")

    baseline_path = METRICS_DIR / f"segmentation_baseline_{split}.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"Loaded baseline mIoU: {baseline['mean_miou']:.3f}")
    else:
        baseline = run_baseline(split, num_images, seed, model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SegFormer on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    distorted: dict = {}
    enhanced: dict = {}

    for distortion in DISTORTION_TYPES:
        distorted[distortion] = {}
        enhanced[distortion] = {}
        for level in default_levels(distortion):
            level_key = str(level)
            rng = np.random.default_rng(level_seed(seed, distortion, level))
            mious_d, mious_e = [], []

            for image_path in tqdm(image_paths, desc=f"seg {distortion}/{level_key}"):
                clean = cv2.imread(str(image_path))
                gt_mask = load_seg_mask(split, image_path.stem)
                if clean is None or gt_mask is None:
                    continue
                dist_img = apply_distortion(clean, distortion, level, rng=rng)
                enh_img = enhance_for_distortion(dist_img, distortion)
                mious_d.append(_eval_image_segmentation(model, processor, device, dist_img, gt_mask)["miou"])
                mious_e.append(_eval_image_segmentation(model, processor, device, enh_img, gt_mask)["miou"])

            distorted[distortion][level_key] = {
                "level": level,
                "mean_miou": float(np.mean(mious_d)),
            }
            enhanced[distortion][level_key] = {
                "level": level,
                "mean_miou": float(np.mean(mious_e)),
            }
            print(
                f"  {distortion} {level_key}: "
                f"dist={distorted[distortion][level_key]['mean_miou']:.3f}  "
                f"enh={enhanced[distortion][level_key]['mean_miou']:.3f}"
            )

    combined = {
        "split": split,
        "num_images": len(image_paths),
        "seed": seed,
        "model": model_id,
        "baseline": {"mean_miou": baseline["mean_miou"]},
        "distorted": distorted,
        "enhanced": enhanced,
    }
    out_json = METRICS_DIR / f"segmentation_robustness_{split}.json"
    out_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    plot_path = FIGURES_DIR / f"segmentation_robustness_{split}.png"
    save_robustness_summary_plot(
        combined,
        plot_path,
        metric_key="mean_miou",
        ylabel="mIoU",
        title="SegFormer mIoU vs distortion intensity",
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {plot_path}")
    return combined


def run_distorted(split: str, num_images: int | None, model_id: str) -> None:
    run_robustness(split, num_images or 10, 42, model_id)


def run_enhanced(split: str, num_images: int | None, model_id: str) -> None:
    run_robustness(split, num_images or 10, 42, model_id)


def run_finetune(split: str, distortion: str, model_id: str) -> None:
    raise NotImplementedError("Implemented in Step 8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegFormer semantic segmentation evaluation.")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=("baseline", "robustness", "distorted", "enhanced", "finetune", "all"),
    )
    parser.add_argument("--distortion", default="noise", choices=("noise", "low_light", "jpeg"))
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
        run_finetune(args.split, args.distortion, args.model)
    elif args.mode == "all":
        run_baseline(args.split, args.num_images, args.seed, args.model)
        run_robustness(args.split, args.num_images, args.seed, args.model)


if __name__ == "__main__":
    main()
