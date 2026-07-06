"""Measure SNR/PSNR between clean and distorted images for every distortion level."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.apply_distortions import list_images
from src.bdd100k_utils import select_paired_images, build_label_index
from src.distortions import apply_distortion, compute_psnr_db, compute_snr_db, default_levels, level_tag
from src.paths import DISTORTION_TYPES, FIGURES_DIR, TABLES_DIR, ensure_output_dirs


def _level_label(distortion: str, level: float | int) -> str:
    if distortion == "noise":
        return f"SNR {int(level)} dB"
    if distortion == "low_light":
        return f"γ = {level:g}"
    if distortion == "jpeg":
        return f"Q = {int(level)}"
    return str(level)


def _severity_note(distortion: str, level: float | int, mean_snr: float, mean_psnr: float) -> str:
    if distortion == "noise":
        return f"Target SNR {level} dB; measured ≈ {mean_snr:.1f} dB"
    if distortion == "low_light":
        return f"Darkening γ={level:g}; measured SNR ≈ {mean_snr:.1f} dB, PSNR ≈ {mean_psnr:.1f} dB"
    return f"JPEG Q={int(level)}; measured SNR ≈ {mean_snr:.1f} dB, PSNR ≈ {mean_psnr:.1f} dB"


def compute_stats(
    split: str = "train",
    num_images: int = 100,
    seed: int = 42,
    *,
    use_paired: bool = True,
) -> list[dict]:
    label_index = build_label_index()
    if use_paired:
        image_paths = select_paired_images(split, num_images, seed, label_index)
    else:
        image_paths = list_images(split, limit=num_images, seed=seed)
    if not image_paths:
        raise SystemExit(f"No images found for split '{split}'.")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for distortion in DISTORTION_TYPES:
        for level in default_levels(distortion):
            snrs: list[float] = []
            psnrs: list[float] = []
            for image_path in image_paths:
                clean = cv2.imread(str(image_path))
                if clean is None:
                    continue
                distorted = apply_distortion(clean, distortion, level, rng=rng)
                snrs.append(compute_snr_db(clean, distorted))
                psnrs.append(compute_psnr_db(clean, distorted))

            if not snrs:
                continue
            rows.append(
                {
                    "distortion": distortion,
                    "parameter_level": level,
                    "level_label": _level_label(distortion, level),
                    "mean_snr_db": float(np.mean(snrs)),
                    "std_snr_db": float(np.std(snrs)),
                    "mean_psnr_db": float(np.mean(psnrs)),
                    "std_psnr_db": float(np.std(psnrs)),
                    "num_images": len(snrs),
                    "visual_severity": _severity_note(
                        distortion, level, float(np.mean(snrs)), float(np.mean(psnrs))
                    ),
                }
            )
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "distortion",
        "parameter_level",
        "level_label",
        "mean_snr_db",
        "std_snr_db",
        "mean_psnr_db",
        "std_psnr_db",
        "num_images",
        "visual_severity",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = [f"{r['distortion'][:3]}/{r['level_label']}" for r in rows]
    x = np.arange(len(labels))

    axes[0].bar(x, [r["mean_snr_db"] for r in rows], color="#3498db", edgecolor="white")
    axes[0].set_title("Measured SNR (clean vs distorted)")
    axes[0].set_ylabel("SNR (dB)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [r["mean_psnr_db"] for r in rows], color="#9b59b6", edgecolor="white")
    axes[1].set_title("Measured PSNR (clean vs distorted)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Distortion severity on robustness evaluation set", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute measured SNR/PSNR per distortion level.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_output_dirs()
    rows = compute_stats(args.split, args.num_images, args.seed)
    csv_path = TABLES_DIR / "distortion_snr_psnr.csv"
    plot_path = FIGURES_DIR / "distortion_snr_psnr.png"
    save_csv(rows, csv_path)
    save_plot(rows, plot_path)
    print(f"Saved {len(rows)} rows to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
