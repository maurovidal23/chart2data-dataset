"""Level and slope changepoints."""

from __future__ import annotations

from typing import Any

import numpy as np


def add_changepoints(
    y: np.ndarray, x_norm: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = y.copy()
    changes: list[dict[str, Any]] = []
    if rng.random() < 0.35:
        index = int(rng.integers(max(2, len(y) // 5), min(len(y) - 1, 4 * len(y) // 5)))
        magnitude = float(rng.uniform(-12, 12))
        result[index:] += magnitude
        changes.append({"type": "level_shift", "index": index, "magnitude": magnitude})
    if rng.random() < 0.3:
        index = int(rng.integers(max(2, len(y) // 5), min(len(y) - 1, 4 * len(y) // 5)))
        slope_delta = float(rng.uniform(-30, 30))
        result[index:] += slope_delta * (x_norm[index:] - x_norm[index])
        changes.append({"type": "slope_change", "index": index, "slope_delta": slope_delta})
    return result, changes
