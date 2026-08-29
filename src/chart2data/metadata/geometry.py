"""Canonical curve resampling and geometry derivation."""

from __future__ import annotations

import numpy as np

from chart2data.rendering.transforms import data_to_pixel
from chart2data.schema import (
    AxisMetadata,
    CurveGrid,
    NormalizedCurve,
    NormalizedPixelCurve,
    PixelCurve,
    PlotArea,
    TransformMetadata,
    VisualResolution,
)


def resample_curve(x: np.ndarray, y: np.ndarray, num_points: int) -> CurveGrid:
    """Linearly resample the rendered polyline over its data-domain extent."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if len(x) != len(y) or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing and aligned with y")
    grid_x = np.linspace(float(x[0]), float(x[-1]), num_points)
    grid_y = np.interp(grid_x, x, y)
    return CurveGrid(num_points=num_points, x=grid_x.tolist(), y=grid_y.tolist())


def normalize_y(values: np.ndarray, axis: AxisMetadata) -> np.ndarray:
    if axis.scale != "linear":
        raise NotImplementedError("v0.1 normalization supports linear axes")
    span = axis.max - axis.min
    if span <= 0:
        raise ValueError("axis maximum must exceed minimum")
    return (np.asarray(values, dtype=float) - axis.min) / span


def denormalize_y(values: np.ndarray, axis: AxisMetadata) -> np.ndarray:
    if axis.scale != "linear":
        raise NotImplementedError("v0.1 normalization supports linear axes")
    return axis.min + np.asarray(values, dtype=float) * (axis.max - axis.min)


def curve_geometry(
    grid: CurveGrid,
    y_axis: AxisMetadata,
    plot_area: PlotArea,
    transform: TransformMetadata,
) -> tuple[NormalizedCurve, PixelCurve, NormalizedPixelCurve]:
    points = np.column_stack([grid.x, grid.y])
    pixels = data_to_pixel(points, transform)
    normalized_y = normalize_y(np.asarray(grid.y), y_axis)
    plot_width = plot_area.right_px - plot_area.left_px
    plot_height = plot_area.bottom_px - plot_area.top_px
    normalized_px_x = (pixels[:, 0] - plot_area.left_px) / plot_width
    normalized_px_y = (pixels[:, 1] - plot_area.top_px) / plot_height
    return (
        NormalizedCurve(x=grid.x.copy(), z=normalized_y.tolist()),
        PixelCurve(x_px=pixels[:, 0].tolist(), y_px=pixels[:, 1].tolist()),
        NormalizedPixelCurve(x=normalized_px_x.tolist(), y=normalized_px_y.tolist()),
    )


def visual_resolution(
    x_axis: AxisMetadata, y_axis: AxisMetadata, plot_area: PlotArea
) -> VisualResolution:
    width = plot_area.right_px - plot_area.left_px
    height = plot_area.bottom_px - plot_area.top_px
    return VisualResolution(
        plot_width_px=width,
        plot_height_px=height,
        x_units_per_pixel=(x_axis.max - x_axis.min) / width,
        y_units_per_pixel=(y_axis.max - y_axis.min) / height,
    )
