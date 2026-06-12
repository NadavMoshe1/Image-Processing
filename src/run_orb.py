"""ORB feature detection and matching evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.distortions import apply_distortion, default_levels
from src.enhancements import enhance_for_distortion
from src.evaluate import orb_matching_ratio, save_orb_summary_plot
from src.paths import (
    BASELINE_FEATURES_DIR,
    DISTORTION_TYPES,
    FIGURES_DIR,
    IMAGES_ROOT,
    METRICS_DIR,
    ensure_output_dirs,
)


def _level_seed(base_seed: int, distortion: str, level: float | int) -> int:
    key = f"{base_seed}:{distortion}:{level}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def select_images(split: str, num_images: int, seed: int) -> list[Path]:
    images = sorted((IMAGES_ROOT / split).glob("*.jpg"))
    if not images:
        return []
    n = min(num_images, len(images))
    rng = random.Random(seed)
    return sorted(rng.sample(images, n))


def _mean_metric(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return float(np.mean([r[key] for r in records]))


def evaluate_on_images(
    image_paths: list[Path],
    query_fn,
    seed: int,
) -> list[dict]:
    """Run ORB matching: clean reference vs query image from query_fn(clean, path)."""
    records = []
    for path in tqdm(image_paths, desc="ORB matching"):
        clean = cv2.imread(str(path))
        if clean is None:
            continue
        query = query_fn(clean, path)
        if query is None:
            continue
        metrics = orb_matching_ratio(clean, query)
        records.append({"image": path.name, **metrics})
    return records


def run_baseline(split: str, image_paths: list[Path], seed: int) -> dict:
    """Clean vs clean — sanity check (expect ratio ~1.0)."""

    def query_fn(clean: np.ndarray, _path: Path) -> np.ndarray:
        return clean

    records = evaluate_on_images(image_paths, query_fn, seed)
    summary = {
        "mean_matching_ratio": _mean_metric(records, "matching_ratio"),
        "per_image": records,
    }
    out = BASELINE_FEATURES_DIR / f"orb_baseline_{split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Baseline mean matching ratio: {summary['mean_matching_ratio']:.4f}")
    return summary


def run_distorted(split: str, image_paths: list[Path], seed: int) -> dict:
    """Clean vs distorted at all intensity levels."""
    summary: dict = {}

    for distortion in DISTORTION_TYPES:
        summary[distortion] = {}
        for level in default_levels(distortion):
            level_key = str(level)
            rng = np.random.default_rng(_level_seed(seed, distortion, level))

            def query_fn(
                clean: np.ndarray,
                _path: Path,
                d=distortion,
                lv=level,
                r=rng,
            ) -> np.ndarray:
                return apply_distortion(clean, d, lv, rng=r)

            records = evaluate_on_images(image_paths, query_fn, seed)
            entry: dict = {
                "level": level,
                "mean_matching_ratio": _mean_metric(records, "matching_ratio"),
                "per_image": records,
            }
            summary[distortion][level_key] = entry
            print(
                f"  {distortion} level={level}: "
                f"matching ratio={entry['mean_matching_ratio']:.4f}"
            )

    out = METRICS_DIR / f"orb_distorted_{split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_enhanced(split: str, image_paths: list[Path], seed: int) -> dict:
    """Clean vs enhanced distorted images at all intensity levels."""
    summary: dict = {}

    for distortion in DISTORTION_TYPES:
        summary[distortion] = {}
        for level in default_levels(distortion):
            level_key = str(level)
            rng = np.random.default_rng(_level_seed(seed, distortion, level))

            def query_fn(
                clean: np.ndarray,
                _path: Path,
                d=distortion,
                lv=level,
                r=rng,
            ) -> np.ndarray:
                distorted = apply_distortion(clean, d, lv, rng=r)
                return enhance_for_distortion(distorted, d)

            records = evaluate_on_images(image_paths, query_fn, seed)
            entry = {
                "level": level,
                "mean_matching_ratio": _mean_metric(records, "matching_ratio"),
                "per_image": records,
            }
            summary[distortion][level_key] = entry
            print(
                f"  {distortion} level={level} (enhanced): "
                f"matching ratio={entry['mean_matching_ratio']:.4f}"
            )

    out = METRICS_DIR / f"orb_enhanced_{split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_combined_results(
    baseline: dict,
    distorted: dict,
    enhanced: dict,
    split: str,
    num_images: int,
    seed: int,
) -> dict:
    return {
        "split": split,
        "num_images": num_images,
        "seed": seed,
        "baseline": baseline,
        "distorted": distorted,
        "enhanced": enhanced,
    }


def save_markdown_table(results: dict, output_path: Path) -> None:
    """Write a markdown table for README."""
    lines = [
        "| Distortion | Level | Distorted | Enhanced | Recovery |",
        "|------------|-------|-----------|----------|----------|",
    ]
    baseline = results["baseline"]["mean_matching_ratio"]

    for distortion in DISTORTION_TYPES:
        dist_levels = results["distorted"].get(distortion, {})
        enh_levels = results["enhanced"].get(distortion, {})
        for level_key in dist_levels:
            d_ratio = dist_levels[level_key]["mean_matching_ratio"]
            e_ratio = enh_levels.get(level_key, {}).get("mean_matching_ratio", d_ratio)
            recovery = e_ratio - d_ratio
            level = dist_levels[level_key]["level"]
            if distortion == "noise":
                label = f"SNR {level} dB"
            elif distortion == "low_light":
                label = f"gamma {level}"
            else:
                label = f"quality {level}"
            lines.append(
                f"| {distortion} | {label} | {d_ratio:.3f} | {e_ratio:.3f} | {recovery:+.3f} |"
            )

    lines.append("")
    lines.append(f"*Clean baseline matching ratio: **{baseline:.3f}***")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORB feature matching evaluation.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        default="all",
        choices=("baseline", "distorted", "enhanced", "all"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    random.seed(args.seed)

    image_paths = select_images(args.split, args.num_images, args.seed)
    if not image_paths:
        raise SystemExit(f"No images found for split '{args.split}'.")

    print(f"Evaluating ORB on {len(image_paths)} images (split={args.split}, seed={args.seed})")

    baseline = distorted = enhanced = {}

    if args.mode in ("baseline", "all"):
        print("\n--- Baseline (clean vs clean) ---")
        baseline = run_baseline(args.split, image_paths, args.seed)

    if args.mode in ("distorted", "all"):
        print("\n--- Distorted ---")
        distorted = run_distorted(args.split, image_paths, args.seed)

    if args.mode in ("enhanced", "all"):
        print("\n--- Enhanced ---")
        enhanced = run_enhanced(args.split, image_paths, args.seed)

    if args.mode == "all":
        combined = build_combined_results(
            baseline, distorted, enhanced, args.split, len(image_paths), args.seed
        )
        combined_path = METRICS_DIR / f"orb_results_{args.split}.json"
        combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

        plot_path = FIGURES_DIR / f"orb_matching_{args.split}.png"
        save_orb_summary_plot(combined, plot_path)

        table_path = METRICS_DIR / f"orb_results_{args.split}.md"
        save_markdown_table(combined, table_path)

        print(f"\nResults:  {combined_path}")
        print(f"Plot:     {plot_path}")
        print(f"Table:    {table_path}")


if __name__ == "__main__":
    main()
