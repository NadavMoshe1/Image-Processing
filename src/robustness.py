"""Shared helpers for distortion robustness evaluation."""

from __future__ import annotations

import hashlib


def level_seed(base_seed: int, distortion: str, level: float | int) -> int:
    key = f"{base_seed}:{distortion}:{level}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def sorted_level_keys(distortion: str, keys: list[str]) -> list[str]:
    if distortion == "jpeg":
        return sorted(keys, key=int, reverse=True)
    return sorted(keys, key=float, reverse=True)
