"""Build SegFormer fine-tuning datasets from distorted BDD100K images + clean seg masks."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.bdd100k_utils import (
    SEG_CLASS_NAMES,
    load_seg_color,
    overlay_seg_mask,
    seg_id_path,
    select_paired_images,
)
from src.distortions import apply_distortion, level_tag
from src.paths import FIGURES_DIR, SEG_DATASET_ROOT, ensure_output_dirs
from src.robustness import level_seed
from src.yolo_dataset import default_finetune_level


def dataset_root(distortion: str, level: float | int) -> Path:
    return SEG_DATASET_ROOT / f"{distortion}_{level_tag(distortion, level)}"


def manifest_path(distortion: str, level: float | int) -> Path:
    return dataset_root(distortion, level) / "manifest.json"


def load_manifest(distortion: str, level: float | int) -> dict:
    path = manifest_path(distortion, level)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _export_split(
    image_paths: list[Path],
    split_name: str,
    source_split: str,
    root: Path,
    distortion: str,
    level: float | int,
    rng: np.random.Generator,
) -> list[str]:
    img_dir = root / "images" / split_name
    mask_dir = root / "masks" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    stems: list[str] = []
    for image_path in tqdm(image_paths, desc=f"Seg {split_name}"):
        clean = cv2.imread(str(image_path))
        if clean is None:
            continue

        stem = image_path.stem
        gt_mask_path = seg_id_path(source_split, stem)
        if not gt_mask_path.exists():
            continue

        distorted = apply_distortion(clean, distortion, level, rng=rng)
        out_img = img_dir / f"{stem}.jpg"
        out_mask = mask_dir / f"{stem}_train_id.png"
        cv2.imwrite(str(out_img), distorted)
        shutil.copy2(gt_mask_path, out_mask)
        stems.append(stem)
    return stems


def build_distorted_seg_dataset(
    distortion: str,
    level: float | int,
    *,
    split: str = "train",
    num_train: int = 300,
    num_val: int = 50,
    seed: int = 42,
    rebuild: bool = False,
) -> Path:
    """
    Export distorted BDD100K images with unchanged semantic segmentation masks.

    Uses the same image pool requirement as robustness eval: detection labels +
    segmentation masks must both exist.
    """
    ensure_output_dirs()
    root = dataset_root(distortion, level)
    manifest_file = manifest_path(distortion, level)
    if manifest_file.exists() and not rebuild:
        print(f"Dataset already exists: {root} (use --rebuild to recreate)")
        return root

    pool_size = num_train + num_val
    pool = select_paired_images(split, pool_size, seed)
    if len(pool) < pool_size:
        print(
            f"Warning: only {len(pool)} images with detection + seg GT "
            f"(requested {pool_size})."
        )
    if len(pool) < 2:
        raise SystemExit(f"Not enough paired images in split '{split}'.")

    rng_split = random.Random(seed)
    shuffled = pool.copy()
    rng_split.shuffle(shuffled)
    n_val = min(num_val, max(1, len(shuffled) - 1))
    n_train = min(num_train, len(shuffled) - n_val)
    train_paths = sorted(shuffled[:n_train])
    val_paths = sorted(shuffled[n_train : n_train + n_val])

    if root.exists() and rebuild:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    noise_rng = np.random.default_rng(level_seed(seed, distortion, level))
    train_stems = _export_split(
        train_paths, "train", split, root, distortion, level, noise_rng
    )
    val_stems = _export_split(
        val_paths, "val", split, root, distortion, level, noise_rng
    )

    manifest = {
        "task": "segmentation",
        "distortion": distortion,
        "level": level,
        "level_tag": level_tag(distortion, level),
        "seed": seed,
        "source_split": split,
        "num_train": len(train_stems),
        "num_val": len(val_stems),
        "num_classes": len(SEG_CLASS_NAMES),
        "class_names": SEG_CLASS_NAMES,
        "ignore_label": 255,
        "train": train_stems,
        "val": val_stems,
        "images_dir": "images",
        "masks_dir": "masks",
        "mask_suffix": "_train_id.png",
        "dataset_root": str(root.resolve()),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built segmentation dataset: {root}")
    print(f"  train={len(train_stems)}  val={len(val_stems)}")
    return root


def save_seg_preview(
    distortion: str,
    level: float | int,
    num_samples: int = 3,
) -> Path:
    """Visualize distorted images with GT segmentation overlays."""
    ensure_output_dirs()
    manifest = load_manifest(distortion, level)
    root = dataset_root(distortion, level)
    stems = manifest["train"][:num_samples]
    if not stems:
        raise SystemExit("No train images in manifest.")

    n = len(stems)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, stem in enumerate(stems):
        img = cv2.imread(str(root / "images" / "train" / f"{stem}.jpg"))
        gt_color = load_seg_color(manifest["source_split"], stem)
        overlay = overlay_seg_mask(img, gt_color) if gt_color is not None else img
        axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, 0].axis("off")
        axes[row, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 0].set_title("Distorted image", fontsize=11)
            axes[row, 1].set_title("GT segmentation", fontsize=11)
        axes[row, 0].set_ylabel(stem, rotation=90, labelpad=36, fontsize=8)

    tag = level_tag(distortion, level)
    fig.suptitle(f"Segmentation dataset preview — {distortion} ({tag})", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / f"seg_dataset_preview_{distortion}_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview: {out}")
    return out


def load_seg_mask_from_dataset(root: Path, split: str, stem: str) -> np.ndarray | None:
    path = root / "masks" / split / f"{stem}_train_id.png"
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SegFormer fine-tuning dataset from distorted BDD100K images."
    )
    parser.add_argument("--distortion", default="noise", choices=("noise", "low_light", "jpeg"))
    parser.add_argument("--level", type=float, default=None, help="Distortion intensity (default: mid level)")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build default mid-level datasets for noise, low_light, and jpeg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distortions = ("noise", "low_light", "jpeg") if args.all else (args.distortion,)

    for distortion in distortions:
        level = args.level if args.level is not None else default_finetune_level(distortion)
        if distortion == "jpeg":
            level = int(level)

        build_distorted_seg_dataset(
            distortion,
            level,
            split=args.split,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
            rebuild=args.rebuild,
        )
        if args.preview:
            save_seg_preview(distortion, level)


if __name__ == "__main__":
    main()
