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
from src.distortions import apply_distortion, default_levels, level_tag
from src.enhancements import enhance_for_distortion, enhancement_label
from src.evaluate import (
    compute_miou,
    load_segmentation_clean_baseline_miou,
    load_segmentation_finetune_batch_results,
    new_segmentation_class_accumulator,
    per_class_miou_from_accumulator,
    save_comparison_bars,
    save_per_class_miou_chart,
    save_robustness_summary_plot,
    save_segformer_training_curves,
    save_segmentation_finetune_summary_gain_plot,
    save_segmentation_finetune_summary_recall_plot,
    save_segmentation_finetune_summary_table_plot,
    update_segmentation_class_accumulator,
)
from src.paths import (
    BASELINE_SEGMENTATION_DIR,
    DISTORTION_TYPES,
    FIGURES_DIR,
    FINETUNE_DIR,
    METRICS_DIR,
    ensure_output_dirs,
)
from src.robustness import level_seed
from src.seg_dataset import (
    build_distorted_seg_dataset,
    dataset_root,
    load_manifest,
    load_seg_mask_from_dataset,
)
from src.yolo_dataset import default_finetune_level

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
    pred = upsampled.argmax(dim=1)
    if pred.dim() == 3:
        pred = pred[0]
    out = pred.cpu().numpy().astype(np.uint8)
    return np.squeeze(out)


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
            acc_d = new_segmentation_class_accumulator()
            acc_e = new_segmentation_class_accumulator()

            for image_path in tqdm(image_paths, desc=f"seg {distortion}/{level_key}"):
                clean = cv2.imread(str(image_path))
                gt_mask = load_seg_mask(split, image_path.stem)
                if clean is None or gt_mask is None:
                    continue
                dist_img = apply_distortion(clean, distortion, level, rng=rng)
                enh_img = enhance_for_distortion(dist_img, distortion)
                res_d = _eval_image_segmentation(model, processor, device, dist_img, gt_mask)
                res_e = _eval_image_segmentation(model, processor, device, enh_img, gt_mask)
                mious_d.append(res_d["miou"])
                mious_e.append(res_e["miou"])
                update_segmentation_class_accumulator(acc_d, res_d["per_class_iou"])
                update_segmentation_class_accumulator(acc_e, res_e["per_class_iou"])

            distorted[distortion][level_key] = {
                "level": level,
                "mean_miou": float(np.mean(mious_d)),
                "per_class_miou": per_class_miou_from_accumulator(acc_d, SEG_CLASS_NAMES),
            }
            enhanced[distortion][level_key] = {
                "level": level,
                "mean_miou": float(np.mean(mious_e)),
                "per_class_miou": per_class_miou_from_accumulator(acc_e, SEG_CLASS_NAMES),
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
    if out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        for key in ("finetuned", "finetuned_eval_note"):
            if key in prev:
                combined[key] = prev[key]
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


def run_build_dataset(
    split: str,
    distortion: str,
    *,
    level: float | int | None = None,
    num_train: int = 300,
    num_val: int = 50,
    seed: int = 42,
    rebuild: bool = False,
) -> Path:
    level = level if level is not None else default_finetune_level(distortion)
    if distortion == "jpeg":
        level = int(level)
    return build_distorted_seg_dataset(
        distortion,
        level,
        split=split,
        num_train=num_train,
        num_val=num_val,
        seed=seed,
        rebuild=rebuild,
    )


class SegFinetuneDataset(torch.utils.data.Dataset):
    """Distorted images + unchanged semantic masks for SegFormer fine-tuning."""

    def __init__(self, root: Path, split: str, stems: list[str], processor) -> None:
        self.root = root
        self.split = split
        self.stems = stems
        self.processor = processor

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        image = cv2.imread(str(self.root / "images" / self.split / f"{stem}.jpg"))
        if image is None:
            raise FileNotFoundError(f"Missing image: {stem}")
        mask = load_seg_mask_from_dataset(self.root, self.split, stem)
        if mask is None:
            raise FileNotFoundError(f"Missing mask: {stem}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoded = self.processor(rgb, mask, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoded.items()}


def _segformer_run_name(distortion: str, level: float | int) -> str:
    return f"segformer_{distortion}_{level_tag(distortion, level)}"


def _compute_segformer_eval_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)
    upsampled = torch.nn.functional.interpolate(
        logits_t,
        size=labels_t.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    preds = upsampled.argmax(dim=1).numpy()
    labels_np = labels_t.numpy()

    mious: list[float] = []
    for pred_mask, gt_mask in zip(preds, labels_np):
        mious.append(compute_miou(pred_mask, gt_mask)["miou"])
    return {"miou": float(np.mean(mious)) if mious else 0.0}


def _load_segformer_model(model_path: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor


def run_finetune(
    split: str,
    distortion: str,
    model_id: str,
    *,
    level: float | int | None = None,
    num_train: int = 300,
    num_val: int = 50,
    seed: int = 42,
    epochs: int = 30,
    batch: int = 2,
    grad_accum: int = 2,
    learning_rate: float = 6e-5,
    rebuild_dataset: bool = False,
) -> dict:
    """Build distorted seg dataset and fine-tune SegFormer."""
    from transformers import Trainer, TrainingArguments

    ensure_output_dirs()
    level = level if level is not None else default_finetune_level(distortion)
    if distortion == "jpeg":
        level = int(level)

    build_distorted_seg_dataset(
        distortion,
        level,
        split=split,
        num_train=num_train,
        num_val=num_val,
        seed=seed,
        rebuild=rebuild_dataset,
    )
    manifest = load_manifest(distortion, level)
    root = dataset_root(distortion, level)
    tag = level_tag(distortion, level)
    run_name = _segformer_run_name(distortion, level)
    output_dir = FINETUNE_DIR / run_name

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id)

    train_ds = SegFinetuneDataset(root, "train", manifest["train"], processor)
    val_ds = SegFinetuneDataset(root, "val", manifest["val"], processor)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="miou",
        greater_is_better=True,
        logging_steps=max(1, len(train_ds) // (batch * grad_accum * 5)),
        fp16=use_cuda,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_segformer_eval_metrics,
    )

    print(f"Fine-tuning SegFormer — {distortion} ({tag}) on {len(train_ds)} train / {len(val_ds)} val...")
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    log_history = trainer.state.log_history
    curves_path = FIGURES_DIR / f"segmentation_finetune_training_{distortion}_{tag}.png"
    save_segformer_training_curves(
        log_history,
        curves_path,
        title=f"SegFormer fine-tuning — {distortion} ({tag})",
    )

    summary = {
        "split": split,
        "distortion": distortion,
        "level": level,
        "seed": seed,
        "num_train": len(manifest["train"]),
        "num_val": len(manifest["val"]),
        "epochs": epochs,
        "batch": batch,
        "grad_accum": grad_accum,
        "base_model": model_id,
        "dataset_root": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_name": run_name,
        "log_history": log_history,
    }
    out_json = METRICS_DIR / f"segmentation_finetune_{distortion}_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Fine-tuning complete. Saved model: {output_dir}")
    print(f"Saved training summary: {out_json}")
    print(f"Saved training curves: {curves_path}")
    return summary


def save_finetune_preview_grid(
    stems: list[str],
    root: Path,
    pretrained_model,
    pretrained_processor,
    finetuned_model,
    finetuned_processor,
    device: str,
    *,
    distortion: str,
    tag: str,
) -> Path:
    """Side-by-side GT / pretrained / fine-tuned segmentation on distorted val images."""
    n = len(stems)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Distorted input", "Ground truth", "Pretrained", "Fine-tuned"]

    for row, stem in enumerate(stems):
        img_path = root / "images" / "val" / f"{stem}.jpg"
        image = cv2.imread(str(img_path))
        gt_mask = load_seg_mask_from_dataset(root, "val", stem)
        if image is None or gt_mask is None:
            continue

        pred_pre = predict_segmentation(pretrained_model, pretrained_processor, image, device)
        pred_ft = predict_segmentation(finetuned_model, finetuned_processor, image, device)
        m_pre = compute_miou(pred_pre, gt_mask)["miou"]
        m_ft = compute_miou(pred_ft, gt_mask)["miou"]

        gt_color = mask_to_color(gt_mask)
        panels = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay_seg_mask(image, gt_color), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay_seg_mask(image, mask_to_color(pred_pre)), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay_seg_mask(image, mask_to_color(pred_ft)), cv2.COLOR_BGR2RGB),
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(
                    f"{stem}\nmIoU: {m_pre:.2f} → {m_ft:.2f}",
                    rotation=90,
                    labelpad=40,
                    fontsize=8,
                )

    fig.suptitle(f"Fine-tuning segmentation preview — {distortion} ({tag})", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / f"segmentation_finetune_preview_{distortion}_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview grid: {out}")
    return out


def run_finetune_plots(
    distortion: str,
    level: float | int | None = None,
    *,
    num_preview: int = 3,
    model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
) -> None:
    """Generate training curves and qualitative preview figures for a seg fine-tune run."""
    ensure_output_dirs()
    level = level if level is not None else default_finetune_level(distortion)
    if distortion == "jpeg":
        level = int(level)

    tag = level_tag(distortion, level)
    run_name = _segformer_run_name(distortion, level)
    output_dir = FINETUNE_DIR / run_name
    train_json = METRICS_DIR / f"segmentation_finetune_{distortion}_{tag}.json"

    if train_json.exists():
        log_history = json.loads(train_json.read_text(encoding="utf-8")).get("log_history", [])
        curves_path = FIGURES_DIR / f"segmentation_finetune_training_{distortion}_{tag}.png"
        save_segformer_training_curves(
            log_history,
            curves_path,
            title=f"SegFormer fine-tuning — {distortion} ({tag})",
        )
        print(f"Saved training curves: {curves_path}")
    elif not (FIGURES_DIR / f"segmentation_finetune_training_{distortion}_{tag}.png").exists():
        print("Skipping training curves — no training log found.")

    if not output_dir.exists():
        print("Skipping preview grid — fine-tuned model not found.")
        return

    manifest = load_manifest(distortion, level)
    root = dataset_root(distortion, level)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model, pretrained_processor = _load_segformer_model(model_id, device)
    finetuned_model, finetuned_processor = _load_segformer_model(str(output_dir), device)

    eval_json = METRICS_DIR / f"segmentation_finetune_eval_{distortion}_{tag}.json"
    val_stems = manifest["val"]
    if eval_json.exists():
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
        ranked = sorted(
            eval_data.get("per_image", []),
            key=lambda row: row["finetuned_miou"] - row["pretrained_miou"],
            reverse=True,
        )
        preview_stems = [row["image"].replace(".jpg", "") for row in ranked[:num_preview]]
    else:
        preview_stems = val_stems[:num_preview]

    if preview_stems:
        save_finetune_preview_grid(
            preview_stems,
            root,
            pretrained_model,
            pretrained_processor,
            finetuned_model,
            finetuned_processor,
            device,
            distortion=distortion,
            tag=tag,
        )


def run_finetune_eval(
    split: str,
    distortion: str,
    model_id: str,
    *,
    level: float | int | None = None,
    seed: int = 42,
) -> dict:
    """Compare pretrained vs enhanced vs fine-tuned mIoU on distorted val images."""
    ensure_output_dirs()
    level = level if level is not None else default_finetune_level(distortion)
    if distortion == "jpeg":
        level = int(level)

    manifest = load_manifest(distortion, level)
    run_name = _segformer_run_name(distortion, level)
    finetuned_dir = FINETUNE_DIR / run_name
    if not finetuned_dir.exists():
        raise SystemExit(
            f"Fine-tuned model not found: {finetuned_dir}\n"
            "Run --mode finetune first."
        )

    root = dataset_root(distortion, level)
    val_stems = manifest["val"]
    if not val_stems:
        raise SystemExit("No val images in dataset manifest.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model, pretrained_processor = _load_segformer_model(model_id, device)
    finetuned_model, finetuned_processor = _load_segformer_model(str(finetuned_dir), device)

    pretrained_mious: list[float] = []
    enhanced_mious: list[float] = []
    finetuned_mious: list[float] = []
    per_image: list[dict] = []

    for stem in tqdm(val_stems, desc="seg finetune eval"):
        val_img_path = root / "images" / "val" / f"{stem}.jpg"
        if not val_img_path.exists():
            continue
        distorted = cv2.imread(str(val_img_path))
        gt_mask = load_seg_mask_from_dataset(root, "val", stem)
        if distorted is None or gt_mask is None:
            continue

        enhanced = enhance_for_distortion(distorted, distortion)
        m_pre = _eval_image_segmentation(pretrained_model, pretrained_processor, device, distorted, gt_mask)
        m_enh = _eval_image_segmentation(pretrained_model, pretrained_processor, device, enhanced, gt_mask)
        m_ft = _eval_image_segmentation(finetuned_model, finetuned_processor, device, distorted, gt_mask)

        pretrained_mious.append(m_pre["miou"])
        enhanced_mious.append(m_enh["miou"])
        finetuned_mious.append(m_ft["miou"])
        per_image.append(
            {
                "image": f"{stem}.jpg",
                "pretrained_miou": m_pre["miou"],
                "enhanced_miou": m_enh["miou"],
                "finetuned_miou": m_ft["miou"],
            }
        )

    if not per_image:
        raise SystemExit(
            "No val images evaluated. Check that data/seg_distorted/ contains "
            "images and masks for the requested distortion."
        )

    mean_pretrained = float(np.mean(pretrained_mious))
    mean_enhanced = float(np.mean(enhanced_mious))
    mean_finetuned = float(np.mean(finetuned_mious))
    tag = level_tag(distortion, level)
    summary = {
        "split": split,
        "distortion": distortion,
        "level": level,
        "seed": seed,
        "num_val_images": len(per_image),
        "pretrained_model": model_id,
        "finetuned_dir": str(finetuned_dir),
        "mean_miou_pretrained": mean_pretrained,
        "mean_miou_enhanced": mean_enhanced,
        "mean_miou_finetuned": mean_finetuned,
        "per_image": per_image,
    }
    out_json = METRICS_DIR / f"segmentation_finetune_eval_{distortion}_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = FIGURES_DIR / f"segmentation_finetune_{distortion}_{tag}.png"
    save_comparison_bars(
        ["Pretrained", enhancement_label(distortion), "Fine-tuned"],
        [mean_pretrained, mean_enhanced, mean_finetuned],
        ylabel="mIoU",
        title=f"Segmentation on distorted val — {distortion} ({tag})",
        output_path=plot_path,
        colors=["#95a5a6", "#f39c12", "#2ecc71"],
    )
    print(
        f"Val mIoU — pretrained: {mean_pretrained:.3f}  "
        f"enhanced: {mean_enhanced:.3f}  "
        f"fine-tuned: {mean_finetuned:.3f}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {plot_path}")

    run_finetune_plots(distortion, level, model_id=model_id)
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


def _best_seg_finetune_preview_stem(distortion: str, level: float | int) -> str | None:
    tag = level_tag(distortion, level)
    eval_json = METRICS_DIR / f"segmentation_finetune_eval_{distortion}_{tag}.json"
    if eval_json.exists():
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
        ranked = sorted(
            eval_data.get("per_image", []),
            key=lambda row: row["finetuned_miou"] - row["pretrained_miou"],
            reverse=True,
        )
        if ranked:
            return ranked[0]["image"].replace(".jpg", "")

    manifest = load_manifest(distortion, level)
    val_stems = manifest.get("val", [])
    return val_stems[0] if val_stems else None


def save_finetune_summary_preview(
    *,
    model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    examples: list[tuple[str, float | int]] | None = None,
) -> Path:
    """Qualitative SegFormer fine-tuning examples — one row per distortion type."""
    ensure_output_dirs()
    if examples is None:
        examples = [(distortion, default_finetune_level(distortion)) for distortion in DISTORTION_TYPES]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model, pretrained_processor = _load_segformer_model(model_id, device)
    rows: list[dict] = []

    for distortion, level in examples:
        if distortion == "jpeg":
            level = int(level)
        tag = level_tag(distortion, level)
        root = dataset_root(distortion, level)
        finetuned_dir = FINETUNE_DIR / _segformer_run_name(distortion, level)
        stem = _best_seg_finetune_preview_stem(distortion, level)
        if stem is None or not finetuned_dir.exists():
            continue

        img_path = root / "images" / "val" / f"{stem}.jpg"
        image = cv2.imread(str(img_path))
        gt_mask = load_seg_mask_from_dataset(root, "val", stem)
        if image is None or gt_mask is None:
            continue

        finetuned_model, finetuned_processor = _load_segformer_model(str(finetuned_dir), device)
        pred_pre = predict_segmentation(pretrained_model, pretrained_processor, image, device)
        pred_ft = predict_segmentation(finetuned_model, finetuned_processor, image, device)
        m_pre = compute_miou(pred_pre, gt_mask)["miou"]
        m_ft = compute_miou(pred_ft, gt_mask)["miou"]

        gt_color = mask_to_color(gt_mask)
        rows.append(
            {
                "distortion_label": _distortion_display_name(distortion),
                "label": _finetune_condition_label(distortion, level),
                "stem": stem,
                "panels": [
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(overlay_seg_mask(image, gt_color), cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(overlay_seg_mask(image, mask_to_color(pred_pre)), cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(overlay_seg_mask(image, mask_to_color(pred_ft)), cv2.COLOR_BGR2RGB),
                ],
                "miou_pre": m_pre,
                "miou_ft": m_ft,
            }
        )

    if not rows:
        raise SystemExit("No fine-tuned SegFormer models / val images found for summary preview.")

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Distorted input", "Ground truth", "Pretrained", "Fine-tuned"]
    for row_idx, row in enumerate(rows):
        for col, panel in enumerate(row["panels"]):
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
                    f"{row['stem']}\nmIoU: {row['miou_pre']:.2f} → {row['miou_ft']:.2f}",
                    rotation=90,
                    labelpad=48,
                    fontsize=8,
                )

    fig.suptitle(
        "SegFormer fine-tuning — qualitative examples (largest mIoU gain per type)",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "segmentation_finetune_summary_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary preview: {out}")
    return out


def run_finetune_summary(summary_path: Path | None = None) -> dict:
    """Generate summary figures from SegFormer fine-tune batch results."""
    ensure_output_dirs()
    grouped, batch = load_segmentation_finetune_batch_results(summary_path)

    if not any(grouped.values()):
        raise SystemExit("No SegFormer fine-tune batch results found.")

    subtitle = (
        f"{batch.get('num_train', 500)} train / {batch.get('num_val', 100)} val · "
        f"{batch.get('epochs', '?')} epochs"
    )
    baseline_miou = load_segmentation_clean_baseline_miou("train")
    recall_path = FIGURES_DIR / "segmentation_finetune_summary_recall.png"
    gain_path = FIGURES_DIR / "segmentation_finetune_summary_gain.png"
    table_path = FIGURES_DIR / "segmentation_finetune_summary_table.png"

    save_segmentation_finetune_summary_recall_plot(
        grouped,
        recall_path,
        subtitle=subtitle,
        baseline_miou=baseline_miou,
    )
    save_segmentation_finetune_summary_gain_plot(grouped, gain_path)
    save_segmentation_finetune_summary_table_plot(
        grouped, table_path, baseline_miou=baseline_miou
    )

    try:
        save_finetune_summary_preview()
    except SystemExit as exc:
        print(exc)

    print(f"Saved summary mIoU plot: {recall_path}")
    print(f"Saved summary gain plot: {gain_path}")
    print(f"Saved summary table plot: {table_path}")
    return {"grouped": grouped, "batch": batch}


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
        choices=("baseline", "robustness", "distorted", "enhanced", "build-dataset", "finetune", "finetune-eval", "finetune-plots", "finetune-summary", "all"),
    )
    parser.add_argument("--distortion", default="noise", choices=("noise", "low_light", "jpeg"))
    parser.add_argument(
        "--level",
        type=float,
        default=None,
        help="Distortion intensity for build-dataset / finetune (default: mid level per type)",
    )
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2, help="Per-device batch size (SegFormer; use 2 on 6GB GPU)")
    parser.add_argument("--grad-accum", type=int, default=2)
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
    elif args.mode == "build-dataset":
        run_build_dataset(
            args.split,
            args.distortion,
            level=args.level,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
            rebuild=args.rebuild_dataset,
        )
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
            grad_accum=args.grad_accum,
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
        run_finetune_plots(args.distortion, args.level, model_id=args.model)
    elif args.mode == "finetune-summary":
        run_finetune_summary()
    elif args.mode == "all":
        run_baseline(args.split, args.num_images, args.seed, args.model)
        run_robustness(args.split, args.num_images, args.seed, args.model)


if __name__ == "__main__":
    main()
