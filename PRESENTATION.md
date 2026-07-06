---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- _class: lead -->

# Robustness of Vision Algorithms Under Image Degradations

Digital Image Processing — Course Project

Nadav Moshe · Alon Ron · 2025–2026

---

## The Question

> When dashcam images degrade, do **classical fixes** help — or do we need to **retrain the model**?

- **Distortions:** noise · low light · JPEG
- **Fixes tested:** enhancement (NLM / CLAHE / bilateral) vs fine-tuning
- **Tasks:** ORB · YOLOv8n · SegFormer-b0

---

## Dataset — BDD100K

100 dashcam images with boxes + segmentation masks (seed 42).

![h:430](outputs/figures/eda_samples.png)

---

## Three Distortions, Four Levels Each

| Distortion | Parameter | Levels (mild → severe) |
|------------|-----------|------------------------|
| Gaussian noise | SNR (dB) | 30 · 20 · 10 · 5 |
| Low light | γ | 0.75 · 0.5 · 0.35 · 0.2 |
| JPEG | Quality *Q* | 90 · 50 · 20 · 10 |

![h:330](outputs/figures/distortion_intensity_train.png)

---

## Distorted vs Enhanced

Each distortion gets one matched enhancement (NLM · CLAHE · bilateral).

![h:470](outputs/figures/distortion_preview_train.png)

---

## Clean Baselines

| Task | Metric | Clean score |
|------|--------|-------------|
| ORB matching | Match ratio | **1.000** |
| YOLOv8n | Recall @ IoU 0.5 | **0.319** |
| SegFormer-b0 | Mean IoU | **0.469** |

*Trends on a 100-image subset — not official leaderboard scores.*

---

## ORB — Fragile in Low Light

Worst case γ = 0.2: matching ratio drops to **0.097**.
CLAHE helps low light (+0.089); NLM hurts mild noise.

![h:400](outputs/figures/orb_matching_train.png)

---

## YOLO — Collapses Under Noise

Recall at SNR 5 dB: **0.077** (from 0.319). Enhancement barely helps.

![h:400](outputs/figures/detection_robustness_train.png)

---

## SegFormer — Noise Blurs Boundaries

mIoU at SNR 5 dB: **0.257** (from 0.469). NLM even hurts mild noise.

![h:400](outputs/figures/segmentation_robustness_train.png)

---

## Key Insight

> A cleaner-looking image is **not** a better input for the model.

- **NLM** smooths texture → hurts ORB and SegFormer
- **CLAHE** helps ORB contrast, not deep models
- **Bilateral** can't restore lost JPEG detail

Classical enhancement rarely recovers frozen deep models.

---

## Fine-Tuning Fixes It

Retrain on distorted data (500 train / 100 val per level, 30 epochs).

| Task | SNR 5 dB distorted | **Fine-tuned** |
|------|-------------------:|---------------:|
| YOLO recall | 0.077 | **0.405** |
| SegFormer mIoU | 0.257 | **0.561** |

Measured on the same 100 images as the frozen models.

---

## Fine-Tuning — YOLO

![h:440](outputs/figures/detection_finetune_summary_recall.png)

Fine-tuned recall beats frozen + enhanced at every level.

---

## Fine-Tuning — SegFormer

![h:440](outputs/figures/segmentation_finetune_summary_recall.png)

Fine-tuned mIoU beats frozen + enhanced at every level.

---

## What Works When

| Distortion | Best fix |
|------------|----------|
| Gaussian noise | **Fine-tuning** |
| Low light | CLAHE (ORB) · fine-tuning (deep) |
| JPEG | **Fine-tuning** |

---

## Conclusions

1. Frozen models fail under heavy noise; enhancement barely helps.
2. ORB sometimes gains from CLAHE; NLM often hurts.
3. **Fine-tuning is the reliable fix** for YOLO and SegFormer.
4. Low-level and high-level robustness diverge.

---

<!-- _class: lead -->

## Thank You

[github.com/NadavMoshe1/Image-Processing](https://github.com/NadavMoshe1/Image-Processing)

Nadav Moshe · Alon Ron
