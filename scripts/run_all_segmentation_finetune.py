"""Batch SegFormer fine-tune: all distortion types × all intensity levels."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distortions import default_levels, level_tag  # noqa: E402
from src.paths import METRICS_DIR, ensure_output_dirs  # noqa: E402
from src.run_segmentation import run_finetune, run_finetune_eval  # noqa: E402

PROGRESS_PATH = METRICS_DIR / "seg_finetune_batch_progress.json"
LOG_PATH = METRICS_DIR / "seg_finetune_batch.log"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _save_progress(state: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    PROGRESS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _jobs(distortions: list[str]) -> list[tuple[str, float | int]]:
    jobs: list[tuple[str, float | int]] = []
    for distortion in distortions:
        for level in default_levels(distortion):
            jobs.append((distortion, level))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SegFormer fine-tune across all distortion levels.")
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-val", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2, help="Per-device batch (2 recommended on 6GB GPU)")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    )
    parser.add_argument("--no-rebuild-dataset", action="store_true", help="Reuse existing seg folders if present")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip train/eval if finetune-eval metrics JSON already exists",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    jobs = _jobs(["noise", "low_light", "jpeg"])
    total = len(jobs)

    state = {
        "status": "running",
        "total_jobs": total,
        "completed_jobs": 0,
        "current_job": None,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "epochs": args.epochs,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "results": [],
        "errors": [],
    }
    _save_progress(state)
    _log(
        f"Starting batch: {total} jobs ({args.num_train} train / {args.num_val} val, "
        f"{args.epochs} epochs, batch={args.batch})"
    )

    for idx, (distortion, level) in enumerate(jobs, start=1):
        tag = level_tag(distortion, level)
        job_name = f"{distortion}/{tag}"
        eval_json = METRICS_DIR / f"segmentation_finetune_eval_{distortion}_{tag}.json"

        state["current_job"] = {
            "index": idx,
            "total": total,
            "distortion": distortion,
            "level": level,
            "tag": tag,
            "phase": "starting",
        }
        _save_progress(state)
        _log(f"=== Job {idx}/{total}: {job_name} ===")

        if args.skip_existing and eval_json.exists():
            _log(f"Skip (eval exists): {eval_json.name}")
            state["completed_jobs"] = idx
            state["results"].append({"job": job_name, "skipped": True})
            _save_progress(state)
            continue

        t0 = time.time()
        try:
            state["current_job"]["phase"] = "build+train"
            _save_progress(state)
            _log("Building dataset + training...")
            run_finetune(
                args.split,
                distortion,
                args.model,
                level=level,
                num_train=args.num_train,
                num_val=args.num_val,
                seed=args.seed,
                epochs=args.epochs,
                batch=args.batch,
                grad_accum=args.grad_accum,
                rebuild_dataset=not args.no_rebuild_dataset,
            )

            state["current_job"]["phase"] = "eval"
            _save_progress(state)
            _log("Running finetune-eval...")
            summary = run_finetune_eval(
                args.split,
                distortion,
                args.model,
                level=level,
                seed=args.seed,
            )

            elapsed = time.time() - t0
            result = {
                "job": job_name,
                "elapsed_sec": round(elapsed, 1),
                "mean_miou_pretrained": summary["mean_miou_pretrained"],
                "mean_miou_enhanced": summary["mean_miou_enhanced"],
                "mean_miou_finetuned": summary["mean_miou_finetuned"],
            }
            state["results"].append(result)
            state["completed_jobs"] = idx
            _log(
                f"Done {job_name} in {elapsed/60:.1f} min — "
                f"pre={summary['mean_miou_pretrained']:.3f} "
                f"enh={summary['mean_miou_enhanced']:.3f} "
                f"ft={summary['mean_miou_finetuned']:.3f}"
            )
        except Exception as exc:
            err = {"job": job_name, "error": str(exc), "traceback": traceback.format_exc()}
            state["errors"].append(err)
            state["completed_jobs"] = idx
            _log(f"FAILED {job_name}: {exc}")
            _log(traceback.format_exc())

        state["current_job"] = None
        _save_progress(state)

    state["status"] = "completed" if not state["errors"] else "completed_with_errors"
    _save_progress(state)

    summary_path = METRICS_DIR / "seg_finetune_batch_summary.json"
    summary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _log(f"Batch finished. Summary: {summary_path}")
    _log(f"Progress log: {PROGRESS_PATH}")
    _log(f"Text log: {LOG_PATH}")

    try:
        from src.run_segmentation import run_finetune_summary

        _log("Generating batch summary figures...")
        run_finetune_summary(summary_path)
    except Exception as exc:
        _log(f"Summary plot generation failed: {exc}")


if __name__ == "__main__":
    main()
