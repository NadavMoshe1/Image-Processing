# Image Processing / Vision — Course Project

**Objective:** Evaluate the robustness of image processing and vision algorithms under image distortions, and measure recovery via pre-processing enhancements and model fine-tuning.

## Project Decisions


| Category                                              | Choice                              | Details                                                                           | Link / Notes                                |
| ----------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------- |
| **Dataset**                                           | BDD100K                             | Driving-scene dataset with bounding-box and semantic-segmentation ground truth    | [BDD100K](https://www.bdd100k.com/)         |
| **Task 1 — Object Detection** (High-level)            | YOLOv8                              | Detect cars, pedestrians, trucks, traffic lights, traffic signs                   | Metric: mAP / Recall                        |
| **Task 2 — Semantic Segmentation** (High-level)       | SegFormer                           | Pixel-wise labels (road, sidewalk, vehicle, sky, etc.)                            | Metric: mIoU                                |
| **Task 3 — Feature / Keypoint Detection** (Low-level) | ORB (OpenCV)                        | Detect and match keypoints between clean and distorted versions of the same image | Metric: Match accuracy / matching ratio     |
| **Distortion 1**                                      | Low light                           | Simulates dark / under-exposed driving conditions                                 | Vary intensity; measure degradation vs. SNR |
| **Distortion 2**                                      | Gaussian noise                      | Simulates sensor / transmission noise                                             | Vary intensity; measure degradation vs. SNR |
| **Distortion 3**                                      | Severe JPEG compression             | Simulates heavy compression artifacts                                             | Vary quality levels (e.g. 90, 50, 20, 10)   |
| **Enhancement — Noise**                               | Non-Local Means (NLM) denoising     | Pre-processing before running models                                              | OpenCV / scikit-image                       |
| **Enhancement — Compression**                         | Interpolation + bilateral filtering | Reduce blocking artifacts                                                         | OpenCV                                      |
| **Enhancement — Low light**                           | CLAHE                               | Contrast Limited Adaptive Histogram Equalization                                  | OpenCV                                      |
| **Fine-tuning**                                       | YOLOv8 (and optionally SegFormer)   | Train on distorted images; evaluate on distorted test set                         | Ultralytics / Hugging Face                  |


## Evaluation Plan

For each task × distortion combination:

1. **Baseline** — measure performance on clean images (ground truth reference)
2. **Distortion** — apply distortions at multiple intensity levels; plot metrics vs. SNR
3. **Enhancement** — run pre-processing on distorted images; measure recovery
4. **Fine-tuning** — fine-tune DL models on distorted data; measure on distorted test images

Results will be reported **per class** and **per distortion intensity** (tables + curves).

## Team


| Name  | Email |
| ----- | ----- |
| *TBD* | *TBD* |


## Data (local — not in git)

Download and unzip from [BDD100K](https://www.bdd100k.com/) into `data/`:

| Path | Contents |
|------|----------|
| `data/bdd100K_images_10k/10k/{train,val,test}/` | 10,000 `.jpg` images |
| `data/bdd100k_label/100k/{train,val,test}/` | 100,000 `.json` label files (detection + lane polylines) |
| `data/bdd100k_seg_maps/labels/{train,val}/` | 8,000 index masks (`*_train_id.png`) |
| `data/bdd100k_seg_maps/color_labels/{train,val}/` | 8,000 color masks (`*_train_color.png`) |
| `bdd100k_drivable_maps/color_labels/train/` | Drivable-area color masks (`*_drivable_color.png`) |

Images and JSON labels match by filename stem (e.g. `0a0a0b1a-7c39d841.jpg` ↔ `0a0a0b1a-7c39d841.json`).

Drivable masks match: `0a0c3694-4cc8b0e3.jpg` ↔ `0a0c3694-4cc8b0e3_drivable_color.png`. Categories (BGR): **direct drivable** (red), **alternative drivable** (blue), **non-drivable** (black). ~908 images in the 10k train subset have drivable GT.

Segmentation masks match the same stem: `7d06fefd-f7be05a6.jpg` ↔ `7d06fefd-f7be05a6_train_color.png`. Segmentation GT is available for **train (7,000)** and **val (1,000)** only — not for the test split.

**Note:** In this download, detection JSON labels overlap with the seg subset for **train only (~3,430 images)**. The val seg images have masks but no matching `box2d` JSON — use the train split for EDA and detection evaluation.

### EDA

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python notebooks/eda.py
python notebooks/eda_drivable.py
```

Outputs:
- `outputs/figures/eda_samples.png` — original, detection GT, semantic seg GT, ORB keypoints
- `outputs/figures/eda_drivable_samples.png` — original + **separate column per drivable category** (direct, alternative, non-drivable)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run all pipeline scripts from the **project root** using `-m`:

```powershell
python -m src.run_orb --help
python -m src.run_detection --help
python -m src.run_segmentation --help

# Step 2 — generate distorted (+ optional enhanced) images
python -m src.apply_distortions --split train --num-images 50 --also-enhance --preview
```

Distorted images are saved under `data/distorted/{distortion}/{level}/{split}/`.  
Enhanced images go to `data/enhanced/{distortion}/{level}/{split}/`.

| Distortion | Levels | Enhancement |
|------------|--------|-------------|
| `noise` | SNR 30, 20, 10, 5 dB | Non-Local Means |
| `low_light` | gamma 0.75, 0.5, 0.35, 0.2 | CLAHE |
| `jpeg` | quality 90, 50, 20, 10 | upscale + bilateral filter |

## Repository Structure

```
data/
  bdd100K_images_10k/   # Raw images (gitignored)
  bdd100k_label/        # Detection JSON labels (gitignored)
  bdd100k_seg_maps/     # Segmentation masks (gitignored)
  distorted/            # Generated distorted images (gitignored)
  enhanced/             # Generated enhanced images (gitignored)
bdd100k_drivable_maps/  # Drivable-area masks (gitignored, supplementary)

src/
  paths.py              # Shared paths and intensity levels
  distortions.py        # Noise, low light, JPEG
  enhancements.py       # NLM, CLAHE, bilateral filter
  apply_distortions.py  # Batch-generate distorted/enhanced datasets
  run_orb.py            # ORB matching evaluation
  run_detection.py      # YOLOv8 evaluation
  run_segmentation.py   # SegFormer evaluation
  evaluate.py           # Shared metrics and plots

outputs/
  baseline/             # Clean-image results (detection, segmentation, features)
  distorted/            # Distorted-image results
  enhanced/             # Post-enhancement results
  metrics/              # CSV / JSON metric tables
  figures/              # Plots and EDA images

notebooks/
  eda.py                # GT visualization (detection, seg, ORB)
  eda_drivable.py       # Drivable-area GT visualization
```

