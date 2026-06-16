"""Generate README figures (distortion/enhancement previews, per-class detection chart)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.apply_distortions import save_enhancement_preview, save_intensity_preview, save_preview
from src.evaluate import save_per_class_miou_chart, save_per_class_recall_chart
from src.paths import FIGURES_DIR, METRICS_DIR, ensure_output_dirs


def main(split: str = "train", preview_seed: int = 99) -> None:
    ensure_output_dirs()

    save_preview(split, preview_seed, FIGURES_DIR / f"distortion_preview_{split}.png")
    save_intensity_preview(split, preview_seed, FIGURES_DIR / f"distortion_intensity_{split}.png")
    save_enhancement_preview(split, preview_seed, FIGURES_DIR / f"enhancement_preview_{split}.png")
    print(f"Saved distortion/enhancement previews to {FIGURES_DIR}")

    baseline_json = METRICS_DIR / f"detection_baseline_{split}.json"
    if baseline_json.exists():
        data = json.loads(baseline_json.read_text(encoding="utf-8"))
        n = data.get("num_images", "?")
        save_per_class_recall_chart(
            data["per_class_recall"],
            output_path=FIGURES_DIR / f"detection_per_class_recall_{split}.png",
            title=f"YOLOv8 per-class recall @ IoU 0.5 — {split} (N={n})",
        )
        print(f"Saved per-class recall chart")
    else:
        print(f"Skip per-class chart — not found: {baseline_json}")

    seg_json = METRICS_DIR / f"segmentation_baseline_{split}.json"
    if seg_json.exists():
        seg = json.loads(seg_json.read_text(encoding="utf-8"))
        n = seg.get("num_images", "?")
        save_per_class_miou_chart(
            seg["per_class_miou"],
            output_path=FIGURES_DIR / f"segmentation_per_class_miou_{split}.png",
            title=f"SegFormer per-class IoU — {split} (N={n})",
            mean_miou=seg.get("mean_miou"),
        )
        print("Saved per-class mIoU chart")
    else:
        print(f"Skip segmentation chart — not found: {seg_json}")


if __name__ == "__main__":
    main()
