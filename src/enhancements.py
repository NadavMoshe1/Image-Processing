"""Image enhancement functions paired with each distortion type."""

from __future__ import annotations

import cv2
import numpy as np

from src.paths import DISTORTION_TYPES

ENHANCEMENT_LABELS = {
    "noise": "NLM",
    "low_light": "CLAHE",
    "jpeg": "Bilateral",
}


def enhancement_label(distortion: str) -> str:
    """Short plot label for the classical restoration paired with a distortion type."""
    try:
        return ENHANCEMENT_LABELS[distortion]
    except KeyError as exc:
        raise ValueError(f"Unknown distortion: {distortion}. Choose from {DISTORTION_TYPES}") from exc


def denoise_nlm(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Non-Local Means denoising (for Gaussian noise)."""
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h,
        h,
        template_window,
        search_window,
    )


def enhance_low_light_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """CLAHE on the L channel in LAB color space (for low light)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def deblock_bilateral_interp(
    image: np.ndarray,
    scale: int = 2,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Upscale with cubic interpolation, bilateral filter, then resize back (for JPEG)."""
    height, width = image.shape[:2]
    upscaled = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    filtered = cv2.bilateralFilter(upscaled, d, sigma_color, sigma_space)
    return cv2.resize(filtered, (width, height), interpolation=cv2.INTER_AREA)


def enhance_for_distortion(image: np.ndarray, distortion: str) -> np.ndarray:
    """Apply the enhancement that matches a given distortion type."""
    if distortion == "noise":
        return denoise_nlm(image)
    if distortion == "low_light":
        return enhance_low_light_clahe(image)
    if distortion == "jpeg":
        return deblock_bilateral_interp(image)
    raise ValueError(f"Unknown distortion: {distortion}. Choose from {DISTORTION_TYPES}")
