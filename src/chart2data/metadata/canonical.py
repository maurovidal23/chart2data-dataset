"""Build immutable canonical sample records from latent and rendered truth."""

from __future__ import annotations

import numpy as np

from chart2data.metadata.geometry import curve_geometry, resample_curve, visual_resolution
from chart2data.rendering.renderer import RenderResult
from chart2data.rendering.transforms import data_to_pixel
from chart2data.schema import CanonicalSample, RenderedPoint, SeriesData


def build_canonical_sample(
    *,
    sample_id: str,
    latent_series_id: str,
    rendering_id: str,
    split: str,
    image_path: str,
    x: np.ndarray,
    y: np.ndarray,
    rendered: RenderResult,
    curve_grid_size: int,
    signal_structure: dict[str, object] | None,
    challenge_group_id: str | None = None,
) -> CanonicalSample:
    source_points = np.column_stack([x, y])
    pixel_points = data_to_pixel(source_points, rendered.transform)
    rendered_points = [
        RenderedPoint(data_x=float(dx), data_y=float(dy), pixel_x=float(px), pixel_y=float(py))
        for (dx, dy), (px, py) in zip(source_points, pixel_points, strict=True)
    ]
    grid = resample_curve(x, y, curve_grid_size)
    normalized, pixel_curve, normalized_pixel = curve_geometry(
        grid, rendered.axes.y, rendered.plot_area, rendered.transform
    )
    return CanonicalSample(
        sample_id=sample_id,
        latent_series_id=latent_series_id,
        rendering_id=rendering_id,
        split=split,
        image_path=image_path,
        challenge_group_id=challenge_group_id,
        data=SeriesData(x=x.tolist(), y=y.tolist()),
        axes=rendered.axes,
        image_geometry=rendered.image_geometry,
        plot_area=rendered.plot_area,
        transform=rendered.transform,
        rendered_points=rendered_points,
        curve_grid=grid,
        normalized_curve=normalized,
        pixel_curve=pixel_curve,
        normalized_pixel_curve=normalized_pixel,
        visual_resolution=visual_resolution(rendered.axes.x, rendered.axes.y, rendered.plot_area),
        style=rendered.style,
        signal_structure=signal_structure,
    )
