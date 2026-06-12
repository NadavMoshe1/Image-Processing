"""ORB feature detection and matching evaluation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.paths import IMAGES_ROOT, ensure_output_dirs


def list_images(split: str, limit: int | None = None) -> list[Path]:
    images = sorted((IMAGES_ROOT / split).glob("*.jpg"))
    if limit is not None:
        return images[:limit]
    return images


def run_baseline(split: str, num_images: int, seed: int) -> None:
    """Run ORB matching baseline on clean images (Step 3)."""
    raise NotImplementedError("Implemented in Step 3")


def run_distorted(split: str, num_images: int, seed: int) -> None:
    """Evaluate ORB matching on distorted images (Step 3)."""
    raise NotImplementedError("Implemented in Step 3")


def run_enhanced(split: str, num_images: int, seed: int) -> None:
    """Evaluate ORB matching after enhancement (Step 3)."""
    raise NotImplementedError("Implemented in Step 3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORB feature matching evaluation.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=("baseline", "distorted", "enhanced", "all"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    random.seed(args.seed)

    runners = {
        "baseline": run_baseline,
        "distorted": run_distorted,
        "enhanced": run_enhanced,
    }

    if args.mode == "all":
        for fn in runners.values():
            fn(args.split, args.num_images, args.seed)
    else:
        runners[args.mode](args.split, args.num_images, args.seed)


if __name__ == "__main__":
    main()
