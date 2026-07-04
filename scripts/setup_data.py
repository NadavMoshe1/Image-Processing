"""Organize BDD100K data layout and refresh YOLO dataset paths for this machine."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _move_if_needed(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    print(f"Moved {src.relative_to(PROJECT_ROOT)} -> {dest.relative_to(PROJECT_ROOT)}")
    return True


def _extract_zip(zip_path: Path, dest_root: Path, marker: Path) -> bool:
    if marker.exists():
        print(f"Already present: {marker.relative_to(PROJECT_ROOT)}")
        return False
    if not zip_path.exists():
        print(f"Skip extract (zip missing): {zip_path.name}")
        return False
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {dest_root.relative_to(PROJECT_ROOT)} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_root)
    print(f"Extracted {zip_path.name}")
    return True


def organize_bdd100k() -> None:
    images_legacy = PROJECT_ROOT / "data" / "10k"
    images_canonical = PROJECT_ROOT / "data" / "bdd100K_images_10k" / "10k"
    _move_if_needed(images_legacy, images_canonical)

    _extract_zip(
        PROJECT_ROOT / "bdd100k_labels.zip",
        PROJECT_ROOT / "data" / "bdd100k_label",
        PROJECT_ROOT / "data" / "bdd100k_label" / "100k" / "train",
    )
    _extract_zip(
        PROJECT_ROOT / "bdd100k_seg_maps.zip",
        PROJECT_ROOT / "data" / "bdd100k_seg_maps",
        PROJECT_ROOT / "data" / "bdd100k_seg_maps" / "labels" / "train",
    )


def consolidate_yolo_distorted(remove_root_copy: bool = True) -> None:
    root_copy = PROJECT_ROOT / "yolo_distorted"
    canonical = PROJECT_ROOT / "data" / "yolo_distorted"

    def _has_datasets(base: Path) -> bool:
        try:
            return (base / "noise_snr_10db" / "manifest.json").exists()
        except OSError:
            return False

    # Remove broken junction/symlink left from an old layout.
    if canonical.exists() and not _has_datasets(canonical):
        try:
            if not any(canonical.iterdir()):
                canonical.rmdir()
            elif canonical.is_symlink() or canonical.is_junction():
                canonical.unlink()
            else:
                shutil.rmtree(canonical)
            print("Removed stale data/yolo_distorted link or empty folder")
        except OSError as exc:
            print(f"Warning: could not clean data/yolo_distorted: {exc}")

    if not root_copy.exists():
        return

    if not canonical.exists() or not _has_datasets(canonical):
        if canonical.exists():
            try:
                shutil.rmtree(canonical)
            except OSError:
                canonical.unlink()
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root_copy), str(canonical))
        print("Moved yolo_distorted -> data/yolo_distorted")
        return

    if remove_root_copy and _has_datasets(canonical):
        if root_copy.is_dir() and not root_copy.is_symlink():
            print("Removing duplicate root yolo_distorted/ (canonical: data/yolo_distorted/)")
            shutil.rmtree(root_copy)


def refresh_yolo_dataset_paths() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.distortions import level_tag
    from src.yolo_dataset import (
        default_finetune_level,
        dataset_root,
        load_manifest,
        write_dataset_yaml,
    )

    for distortion in ("noise", "low_light", "jpeg"):
        level = default_finetune_level(distortion)
        root = dataset_root(distortion, level)
        if not (root / "manifest.json").exists():
            print(f"Skip refresh (missing): {root.relative_to(PROJECT_ROOT)}")
            continue
        yaml_path = write_dataset_yaml(root)
        manifest = load_manifest(distortion, level)
        manifest["dataset_yaml"] = str(yaml_path.resolve())
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Refreshed paths: {root.relative_to(PROJECT_ROOT)}")


def verify_layout() -> int:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.paths import (
        IMAGES_ROOT,
        LABELS_ROOT,
        SEG_COLOR_ROOT,
        SEG_ID_ROOT,
        YOLO_DATASET_ROOT,
    )

    checks: list[tuple[str, Path, str]] = [
        ("train images", IMAGES_ROOT / "train", "*.jpg"),
        ("train labels", LABELS_ROOT / "train", "*.json"),
        ("train seg id masks", SEG_ID_ROOT / "train", "*_train_id.png"),
        ("train seg color masks", SEG_COLOR_ROOT / "train", "*_train_color.png"),
        ("YOLO fine-tune datasets", YOLO_DATASET_ROOT, "noise_snr_10db"),
    ]

    ok = True
    print("\nData layout check:")
    for name, path, pattern in checks:
        if not path.exists():
            print(f"  [MISSING] {name}: {path.relative_to(PROJECT_ROOT)}")
            ok = False
            continue
        if pattern.endswith("db"):
            found = (path / pattern).exists()
            count = 1 if found else 0
        else:
            count = len(list(path.glob(pattern)))
        status = "OK" if count else "EMPTY"
        if not count:
            ok = False
        print(f"  [{status}] {name}: {path.relative_to(PROJECT_ROOT)} ({count})")

    root_dup = PROJECT_ROOT / "yolo_distorted"
    if root_dup.exists():
        print(f"  [WARN] Duplicate folder still present: yolo_distorted/ (use data/yolo_distorted/)")
        ok = False

    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize BDD100K data and refresh dataset paths.")
    parser.add_argument("--skip-extract", action="store_true", help="Only refresh paths / verify")
    parser.add_argument("--keep-root-yolo", action="store_true", help="Do not delete root yolo_distorted/")
    args = parser.parse_args()

    if not args.skip_extract:
        organize_bdd100k()
        consolidate_yolo_distorted(remove_root_copy=not args.keep_root_yolo)

    refresh_yolo_dataset_paths()
    raise SystemExit(verify_layout())


if __name__ == "__main__":
    main()
