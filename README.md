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


## Repository Structure

```
data/           # Raw and distorted images
src/            # Distortion, enhancement, model, and evaluation scripts
outputs/        # Predictions, metrics, figures
notebooks/      # EDA and visualization
```

