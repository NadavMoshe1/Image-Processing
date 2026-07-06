"""Patch seg_finetune_batch_summary.json when a job completed but batch log missed it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distortions import level_tag
from src.paths import METRICS_DIR


def patch_gamma_035() -> None:
    summary_path = METRICS_DIR / "seg_finetune_batch_summary.json"
    eval_path = METRICS_DIR / "segmentation_finetune_eval_low_light_gamma_0.35.json"
    if not summary_path.exists() or not eval_path.exists():
        print("Summary or eval JSON missing; nothing to patch.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    job = "low_light/gamma_0.35"
    tag = level_tag("low_light", 0.35)

    row = {
        "job": job,
        "distortion": "low_light",
        "level": 0.35,
        "tag": tag,
        "mean_miou_pretrained": eval_data["mean_miou_pretrained"],
        "mean_miou_enhanced": eval_data["mean_miou_enhanced"],
        "mean_miou_finetuned": eval_data["mean_miou_finetuned"],
        "gain_finetuned_vs_pretrained": eval_data["mean_miou_finetuned"] - eval_data["mean_miou_pretrained"],
    }

    results = summary.get("results", [])
    results = [r for r in results if r.get("job") != job]
    results.append(row)
    summary["results"] = sorted(results, key=lambda r: r["job"])
    errors = [e for e in summary.get("errors", []) if job not in str(e)]
    summary["errors"] = errors
    if summary.get("completed_with_errors") and not errors:
        summary["completed_with_errors"] = False

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Patched {summary_path} with {job}")


if __name__ == "__main__":
    patch_gamma_035()
