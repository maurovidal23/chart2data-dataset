"""NumPy-only reconstruction metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _aligned(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(predicted, dtype=float)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("actual and predicted must have the same non-empty shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("metric inputs must be finite")
    return left, right


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    left, right = _aligned(actual, predicted)
    return float(np.mean(np.abs(left - right)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    left, right = _aligned(actual, predicted)
    return float(np.sqrt(np.mean((left - right) ** 2)))


def nmae(actual: ArrayLike, predicted: ArrayLike, y_min: float, y_max: float) -> float:
    span = y_max - y_min
    if span <= 0:
        raise ValueError("y_max must exceed y_min")
    return mae(actual, predicted) / span


def pixel_equivalent_error(
    actual: ArrayLike, predicted: ArrayLike, y_min: float, y_max: float, plot_height_px: float
) -> float:
    """Mean absolute y error expressed as expected vertical plot pixels."""
    if plot_height_px <= 0:
        raise ValueError("plot_height_px must be positive")
    return nmae(actual, predicted, y_min, y_max) * plot_height_px


def curve_shape_error(actual_z: ArrayLike, predicted_z: ArrayLike) -> float:
    """MAE on fixed-grid normalized curves."""
    return mae(actual_z, predicted_z)
