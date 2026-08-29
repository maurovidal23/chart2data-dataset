"""Structural and numerical consistency checks."""

from __future__ import annotations

import numpy as np

from chart2data.rendering.transforms import data_to_pixel, pixel_to_data
from chart2data.schema import CanonicalSample


def validate_sample(sample: CanonicalSample, tolerance: float = 1e-7) -> list[str]:
    errors: list[str] = []
    points = np.column_stack([sample.data.x, sample.data.y])
    pixels = data_to_pixel(points, sample.transform)
    recovered = pixel_to_data(pixels, sample.transform)
    if not np.allclose(points, recovered, atol=tolerance, rtol=tolerance):
        errors.append("coordinate transform round trip failed")
    stored = np.array([[point.pixel_x, point.pixel_y] for point in sample.rendered_points])
    if not np.allclose(pixels, stored, atol=tolerance, rtol=tolerance):
        errors.append("rendered point coordinates do not match transform")
    if len(sample.curve_grid.x) != sample.curve_grid.num_points:
        errors.append("curve grid length mismatch")
    if not np.all(
        (np.asarray(sample.normalized_curve.z) >= 0) & (np.asarray(sample.normalized_curve.z) <= 1)
    ):
        errors.append("normalized curve lies outside visible y-axis")
    return errors


def validate_split_leakage(samples: list[CanonicalSample]) -> list[str]:
    assignments: dict[str, set[str]] = {}
    for sample in samples:
        assignments.setdefault(sample.latent_series_id, set()).add(sample.split)
    return [series_id for series_id, splits in assignments.items() if len(splits) > 1]
