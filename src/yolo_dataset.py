"""Build YOLO-format datasets from BDD100K detection labels + distorted images."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from src.bdd100k_utils import (
    BDD_DETECTION_CLASS_NAMES,
    bdd_boxes_to_yolo_lines,
    build_label_index,
    draw_detection_boxes,
    load_detection_boxes,
    resolve_label_path,
    select_detection_images,
)
from src.distortions import apply_distortion, level_tag
from src.paths import FIGURES_DIR, YOLO_DATASET_ROOT, ensure_output_dirs
from src.robustness import level_seed


def default_finetune_level(distortion: str) -> float | int:
    if distortion == "noise":
        return 10
    if distortion == "low_light":
        return 0.35
    if distortion == "jpeg":
        return 20
    raise ValueError(f"Unknown distortion: {distortion}")


def dataset_root(distortion: str, level: float | int) -> Path:
    return YOLO_DATASET_ROOT / f"{distortion}_{level_tag(distortion, level)}"


def manifest_path(distortion: str, level: float | int) -> Path:
    return dataset_root(distortion, level) / "manifest.json"


def load_manifest(distortion: str, level: float | int) -> dict:
    path = manifest_path(distortion, level)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_dataset_yaml(root: Path) -> Path:
    yaml_path = root / "dataset.yaml"
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(BDD_DETECTION_CLASS_NAMES),
        "names": {idx: name for idx, name in enumerate(BDD_DETECTION_CLASS_NAMES)},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def _export_split(
    image_paths: list[Path],
    split_name: str,
    root: Path,
    label_index: dict[str, Path],
    distortion: str,
    level: float | int,
    rng: np.random.Generator,
) -> list[str]:
    img_dir = root / "images" / split_name
    lbl_dir = root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stems: list[str] = []
    for image_path in tqdm(image_paths, desc=f"YOLO {split_name}"):
        clean = cv2.imread(str(image_path))
        if clean is None:
            continue

        label_path = resolve_label_path(image_path.stem, label_index)
        gt_boxes = load_detection_boxes(label_path)
        lines = bdd_boxes_to_yolo_lines(gt_boxes, clean.shape[1], clean.shape[0])
        if not lines:
            continue

        distorted = apply_distortion(clean, distortion, level, rng=rng)
        stem = image_path.stem
        out_img = img_dir / f"{stem}.jpg"
        out_lbl = lbl_dir / f"{stem}.txt"
        cv2.imwrite(str(out_img), distorted)
        out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stems.append(stem)
    return stems


def build_distorted_yolo_dataset(
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
    Export distorted BDD100K images + clean GT boxes in YOLO format.

    Train/val stems are disjoint subsets of `split` images with detection labels.
    """
    ensure_output_dirs()
    root = dataset_root(distortion, level)
    manifest_file = manifest_path(distortion, level)
    if manifest_file.exists() and not rebuild:
        write_dataset_yaml(root)
        print(f"Dataset already exists: {root} (use --rebuild to recreate)")
        return root

    label_index = build_label_index()
    pool_size = num_train + num_val
    pool = select_detection_images(split, pool_size, seed, label_index)
    if len(pool) < pool_size:
        print(
            f"Warning: only {len(pool)} images with detection labels "
            f"(requested {pool_size})."
        )
    if len(pool) < 2:
        raise SystemExit(f"Not enough detection images in split '{split}'.")

    rng_split = random.Random(seed)
    shuffled = pool.copy()
    rng_split.shuffle(shuffled)
    n_val = min(num_val, max(1, len(shuffled) - 1))
    n_train = min(num_train, len(shuffled) - n_val)
    train_paths = sorted(shuffled[:n_train])
    val_paths = sorted(shuffled[n_train : n_train + n_val])

    if root.exists() and rebuild:
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    noise_rng = np.random.default_rng(level_seed(seed, distortion, level))
    train_stems = _export_split(
        train_paths, "train", root, label_index, distortion, level, noise_rng
    )
    val_stems = _export_split(
        val_paths, "val", root, label_index, distortion, level, noise_rng
    )

    yaml_path = write_dataset_yaml(root)
    manifest = {
        "distortion": distortion,
        "level": level,
        "level_tag": level_tag(distortion, level),
        "seed": seed,
        "source_split": split,
        "num_train": len(train_stems),
        "num_val": len(val_stems),
        "train": train_stems,
        "val": val_stems,
        "class_names": BDD_DETECTION_CLASS_NAMES,
        "dataset_yaml": str(yaml_path),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built YOLO dataset: {root}")
    print(f"  train={len(train_stems)}  val={len(val_stems)}  yaml={yaml_path}")
    return root


def save_yolo_preview(
    distortion: str,
    level: float | int,
    num_samples: int = 3,
) -> Path:
    """Visualize distorted images with GT boxes for sanity checking."""
    ensure_output_dirs()
    manifest = load_manifest(distortion, level)
    root = dataset_root(distortion, level)
    stems = manifest["train"][:num_samples]
    if not stems:
        raise SystemExit("No train images in manifest.")

    n = len(stems)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for row, stem in enumerate(stems):
        img = cv2.imread(str(root / "images" / "train" / f"{stem}.jpg"))
        lbl_path = root / "labels" / "train" / f"{stem}.txt"
        boxes = load_yolo_label_boxes(lbl_path, img.shape[1], img.shape[0])
        vis = draw_detection_boxes(img, boxes)
        axes[row].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[row].axis("off")
        axes[row].set_ylabel(stem, rotation=90, labelpad=36, fontsize=8)

    tag = level_tag(distortion, level)
    fig.suptitle(f"YOLO dataset preview — {distortion} ({tag})", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / f"yolo_dataset_preview_{distortion}_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview: {out}")
    return out


def load_yolo_label_boxes(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    if not label_path.exists():
        return []
    boxes: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append(
            {
                "category": BDD_DETECTION_CLASS_NAMES[cls_id],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from distorted BDD100K images.")
    parser.add_argument("--distortion", default="noise", choices=("noise", "low_light", "jpeg"))
    parser.add_argument("--level", type=float, default=None, help="Distortion intensity (default: mid level)")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = args.level if args.level is not None else default_finetune_level(args.distortion)
    if args.distortion == "jpeg":
        level = int(level)

    build_distorted_yolo_dataset(
        args.distortion,
        level,
        split=args.split,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        rebuild=args.rebuild,
    )
    if args.preview:
        save_yolo_preview(args.distortion, level)


if __name__ == "__main__":
    main()
