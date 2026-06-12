"""Image distortion functions: Gaussian noise, low light, JPEG compression."""

from __future__ import annotations

import cv2
import numpy as np

from src.paths import DISTORTION_TYPES, JPEG_QUALITY, LOW_LIGHT_GAMMA, NOISE_SNR_DB


def compute_snr_db(clean: np.ndarray, distorted: np.ndarray) -> float:
    """Compute SNR in dB between clean and distorted images."""
    clean_f = clean.astype(np.float64)
    noise = distorted.astype(np.float64) - clean_f
    signal_power = np.mean(clean_f**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-12:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def add_gaussian_noise(image: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Gaussian noise to reach a target SNR (dB)."""
    rng = rng or np.random.default_rng()
    img = image.astype(np.float64)
    signal_power = np.mean(img**2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power)
    noisy = img + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_low_light(image: np.ndarray, gamma: float) -> np.ndarray:
    """Darken image via gamma correction (smaller gamma → darker)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    normalized = image.astype(np.float64) / 255.0
    # exponent > 1 darkens; gamma=0.5 → exponent=2
    darkened = np.power(normalized, 1.0 / gamma)
    return np.clip(darkened * 255.0, 0, 255).astype(np.uint8)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression at the given quality level (0–100)."""
    quality = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def apply_distortion(
    image: np.ndarray,
    distortion: str,
    level: float | int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a named distortion at a given intensity level."""
    if distortion == "noise":
        return add_gaussian_noise(image, float(level), rng=rng)
    if distortion == "low_light":
        return apply_low_light(image, float(level))
    if distortion == "jpeg":
        return apply_jpeg_compression(image, int(level))
    raise ValueError(f"Unknown distortion: {distortion}. Choose from {DISTORTION_TYPES}")


def default_levels(distortion: str) -> list[float | int]:
    """Return default intensity levels for a distortion type."""
    if distortion == "noise":
        return list(NOISE_SNR_DB)
    if distortion == "low_light":
        return list(LOW_LIGHT_GAMMA)
    if distortion == "jpeg":
        return list(JPEG_QUALITY)
    raise ValueError(f"Unknown distortion: {distortion}")


def level_tag(distortion: str, level: float | int) -> str:
    """Filesystem-safe tag for a distortion intensity level."""
    if distortion == "noise":
        return f"snr_{int(level)}db"
    if distortion == "low_light":
        return f"gamma_{level:g}"
    if distortion == "jpeg":
        return f"q{int(level)}"
    raise ValueError(f"Unknown distortion: {distortion}")
