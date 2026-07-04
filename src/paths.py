"""Shared project paths and experiment constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Data ---
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_ROOT = DATA_DIR / "bdd100K_images_10k" / "10k"
LABELS_ROOT = DATA_DIR / "bdd100k_label" / "100k"
SEG_COLOR_ROOT = DATA_DIR / "bdd100k_seg_maps" / "color_labels"
SEG_ID_ROOT = DATA_DIR / "bdd100k_seg_maps" / "labels"
DRIVABLE_COLOR_ROOT = PROJECT_ROOT / "bdd100k_drivable_maps" / "color_labels"

DISTORTED_DIR = DATA_DIR / "distorted"
ENHANCED_DIR = DATA_DIR / "enhanced"
YOLO_DATASET_ROOT = DATA_DIR / "yolo_distorted"
SEG_DATASET_ROOT = DATA_DIR / "seg_distorted"

# --- Outputs ---
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
BASELINE_DIR = OUTPUTS_DIR / "baseline"
DISTORTED_OUTPUT_DIR = OUTPUTS_DIR / "distorted"
ENHANCED_OUTPUT_DIR = OUTPUTS_DIR / "enhanced"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FINETUNE_DIR = OUTPUTS_DIR / "finetune"

BASELINE_DETECTION_DIR = BASELINE_DIR / "detection"
BASELINE_SEGMENTATION_DIR = BASELINE_DIR / "segmentation"
BASELINE_FEATURES_DIR = BASELINE_DIR / "features"

# --- Distortion intensity levels ---
NOISE_SNR_DB = [30, 20, 10, 5]
JPEG_QUALITY = [90, 50, 20, 10]
LOW_LIGHT_GAMMA = [0.75, 0.5, 0.35, 0.2]

DISTORTION_TYPES = ("noise", "low_light", "jpeg")

# Expected layout under data/ (see scripts/setup_data.py):
#   bdd100K_images_10k/10k/{train,val,test}/*.jpg
#   bdd100k_label/100k/{train,val,test}/*.json
#   bdd100k_seg_maps/labels/{train,val,test}/*_train_id.png
#   bdd100k_seg_maps/color_labels/{train,val,test}/*_train_color.png
#   yolo_distorted/{distortion}_{level_tag}/  (detection fine-tuning exports)
#   seg_distorted/{distortion}_{level_tag}/   (segmentation fine-tuning exports)


def _ensure_dir(path: Path) -> None:
    """Create a directory; recover from a broken junction/symlink on Windows."""
    if path.is_symlink():
        path.unlink()
    elif path.exists() and not path.is_dir():
        path.unlink()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        if not path.is_dir():
            raise


def ensure_output_dirs() -> None:
    """Create standard output directories if they do not exist."""
    for path in (
        FIGURES_DIR,
        BASELINE_DETECTION_DIR,
        BASELINE_SEGMENTATION_DIR,
        BASELINE_FEATURES_DIR,
        DISTORTED_OUTPUT_DIR,
        ENHANCED_OUTPUT_DIR,
        METRICS_DIR,
        DISTORTED_DIR,
        ENHANCED_DIR,
        YOLO_DATASET_ROOT,
        SEG_DATASET_ROOT,
        FINETUNE_DIR,
    ):
        _ensure_dir(path)
