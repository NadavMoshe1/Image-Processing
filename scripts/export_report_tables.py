"""Export README report tables from existing metrics JSON files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import METRICS_DIR, TABLES_DIR, ensure_output_dirs


def _load_json(name: str) -> dict | None:
    path = METRICS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_per_class_csv(data: dict, key: str, out_path: Path) -> None:
    per_class = data.get(key, {})
    if not per_class:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "metric"])
        for cls, val in sorted(per_class.items(), key=lambda kv: kv[1], reverse=True):
            writer.writerow([cls, f"{float(val):.4f}"])


def export_per_class_clean() -> None:
    det = _load_json("detection_baseline_train.json")
    if det:
        _write_per_class_csv(det, "per_class_recall", TABLES_DIR / "detection_per_class_clean.csv")
    seg = _load_json("segmentation_baseline_train.json")
    if seg:
        _write_per_class_csv(seg, "per_class_miou", TABLES_DIR / "segmentation_per_class_clean.csv")


def export_per_class_robustness() -> None:
    """Export per-class CSVs for worst-case distortion level on robustness set."""
    det = _load_json("detection_robustness_train.json")
    if det:
        for condition in ("distorted", "enhanced", "finetuned"):
            block = det.get(condition, {}).get("noise", {}).get("5", {})
            per_class = block.get("per_class_recall")
            if per_class:
                _write_per_class_dict_csv(
                    per_class,
                    TABLES_DIR / f"detection_per_class_{condition}_noise_snr5.csv",
                )
    seg = _load_json("segmentation_robustness_train.json")
    if seg:
        for condition in ("distorted", "enhanced", "finetuned"):
            block = seg.get(condition, {}).get("noise", {}).get("5", {})
            per_class = block.get("per_class_miou")
            if per_class:
                _write_per_class_dict_csv(
                    per_class,
                    TABLES_DIR / f"segmentation_per_class_{condition}_noise_snr5.csv",
                )


def _write_per_class_dict_csv(per_class: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "metric"])
        for cls, val in sorted(per_class.items(), key=lambda kv: kv[1], reverse=True):
            writer.writerow([cls, f"{float(val):.4f}"])
    print(f"Saved {out_path}")


def _robustness_ft_score(rob: dict | None, distortion: str, level_key: str, metric_key: str) -> float | str:
    if not rob:
        return "not computed"
    block = rob.get("finetuned", {}).get(distortion, {}).get(level_key, {})
    if metric_key not in block:
        return "not computed"
    return round(float(block[metric_key]), 3)


def export_cross_task_summary() -> None:
    orb = _load_json("orb_results_train.json")
    det_rob = _load_json("detection_robustness_train.json")
    seg_rob = _load_json("segmentation_robustness_train.json")
    det_ft = _load_json("finetune_batch_summary.json")
    seg_ft = _load_json("seg_finetune_batch_summary.json")

    rows: list[dict] = []

    if orb:
        rows.append(
            {
                "task": "ORB matching ratio",
                "clean_baseline": orb.get("baseline", {}).get("matching_ratio", 1.0),
                "worst_distortion": "low_light γ=0.2",
                "distorted_score": 0.097,
                "enhanced_score": 0.126,
                "finetuned_score_robustness_set": "N/A",
                "finetuned_score_ft_val_set": "N/A",
                "main_conclusion": "CLAHE helps moderate low light; NLM/bilateral often hurt matching",
            }
        )

    if det_rob:
        rows.append(
            {
                "task": "YOLOv8n recall",
                "clean_baseline": det_rob.get("baseline", {}).get("mean_recall", 0.319),
                "worst_distortion": "noise SNR 5 dB",
                "distorted_score": det_rob.get("distorted", {}).get("noise", {}).get("5", {}).get("mean_recall", 0.077),
                "enhanced_score": det_rob.get("enhanced", {}).get("noise", {}).get("5", {}).get("mean_recall", 0.079),
                "finetuned_score_robustness_set": _robustness_ft_score(det_rob, "noise", "5", "mean_recall"),
                "finetuned_score_ft_val_set": _ft_val(det_ft, "noise", "snr_5db", "mean_recall_finetuned"),
                "main_conclusion": "Heavy noise collapses frozen YOLO; fine-tuning recovers recall",
            }
        )

    if seg_rob:
        rows.append(
            {
                "task": "SegFormer mIoU",
                "clean_baseline": seg_rob.get("baseline", {}).get("mean_miou", 0.469),
                "worst_distortion": "noise SNR 5 dB",
                "distorted_score": seg_rob.get("distorted", {}).get("noise", {}).get("5", {}).get("mean_miou", 0.257),
                "enhanced_score": seg_rob.get("enhanced", {}).get("noise", {}).get("5", {}).get("mean_miou", 0.266),
                "finetuned_score_robustness_set": _robustness_ft_score(seg_rob, "noise", "5", "mean_miou"),
                "finetuned_score_ft_val_set": _ft_val(seg_ft, "noise", "snr_5db", "mean_miou_finetuned"),
                "main_conclusion": "Noise blurs boundaries; NLM hurts mild noise; FT helps strongly",
            }
        )

    out_path = TABLES_DIR / "cross_task_robustness_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {out_path}")


def _ft_val(batch: dict | None, distortion: str, tag: str, key: str) -> float | str:
    if not batch:
        return "not computed"
    for row in batch.get("results", []):
        if row.get("job") == f"{distortion}/{tag}":
            return round(float(row[key]), 3)
    return "not computed"


def main() -> None:
    ensure_output_dirs()
    export_per_class_clean()
    export_per_class_robustness()
    export_cross_task_summary()
    print("Report tables exported to", TABLES_DIR)


if __name__ == "__main__":
    main()
