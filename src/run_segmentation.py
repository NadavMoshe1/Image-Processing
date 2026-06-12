"""SegFormer semantic segmentation baseline and evaluation."""

from __future__ import annotations

import argparse

from src.paths import BASELINE_SEGMENTATION_DIR, ensure_output_dirs


def run_baseline(split: str, num_images: int | None, model_id: str) -> None:
    """Run pretrained SegFormer on clean images and compute mIoU."""
    raise NotImplementedError("Implemented in Step 5")


def run_distorted(split: str, num_images: int | None, model_id: str) -> None:
    """Evaluate SegFormer on distorted images."""
    raise NotImplementedError("Implemented in Step 6")


def run_enhanced(split: str, num_images: int | None, model_id: str) -> None:
    """Evaluate SegFormer on enhanced distorted images."""
    raise NotImplementedError("Implemented in Step 7")


def run_finetune(split: str, distortion: str, model_id: str) -> None:
    """Fine-tune SegFormer on distorted training images."""
    raise NotImplementedError("Implemented in Step 8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegFormer semantic segmentation evaluation.")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--num-images", type=int, default=None, help="Limit images (None = all)")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=("baseline", "distorted", "enhanced", "finetune", "all"),
    )
    parser.add_argument("--distortion", default="noise", choices=("noise", "low_light", "jpeg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    if args.mode == "baseline":
        run_baseline(args.split, args.num_images, args.model)
    elif args.mode == "distorted":
        run_distorted(args.split, args.num_images, args.model)
    elif args.mode == "enhanced":
        run_enhanced(args.split, args.num_images, args.model)
    elif args.mode == "finetune":
        run_finetune(args.split, args.distortion, args.model)
    else:
        run_baseline(args.split, args.num_images, args.model)
        run_distorted(args.split, args.num_images, args.model)
        run_enhanced(args.split, args.num_images, args.model)


if __name__ == "__main__":
    main()
