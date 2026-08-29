"""Canonical, prediction-agnostic metadata models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeriesData(StrictModel):
    x: list[float]
    y: list[float]

    @model_validator(mode="after")
    def matching_lengths(self) -> SeriesData:
        if len(self.x) != len(self.y) or len(self.x) < 2:
            raise ValueError("x and y must have matching lengths >= 2")
        return self


class AxisMetadata(StrictModel):
    scale: Literal["linear", "log", "date", "categorical"] = "linear"
    min: float
    max: float
    ticks: list[float]
    tick_labels: list[str]
    label: str | None = None
    unit: str | None = None
    formatter: str | None = None


class AxesMetadata(StrictModel):
    x: AxisMetadata
    y: AxisMetadata


class ImageGeometry(StrictModel):
    width_px: int
    height_px: int
    dpi: int


class PlotArea(StrictModel):
    left_px: float
    right_px: float
    top_px: float
    bottom_px: float


class RenderedPoint(StrictModel):
    data_x: float
    data_y: float
    pixel_x: float
    pixel_y: float


class CurveGrid(StrictModel):
    num_points: int
    x: list[float]
    y: list[float]


class NormalizedCurve(StrictModel):
    x: list[float]
    z: list[float]
    convention: str = "z=0 bottom, z=1 top"


class PixelCurve(StrictModel):
    x_px: list[float]
    y_px: list[float]


class NormalizedPixelCurve(StrictModel):
    x: list[float]
    y: list[float]
    convention: str = "plot-relative: (0,0) top-left, (1,1) bottom-right"


class VisualResolution(StrictModel):
    plot_width_px: float
    plot_height_px: float
    x_units_per_pixel: float
    y_units_per_pixel: float


class TransformMetadata(StrictModel):
    coordinate_system: str = "image pixels, origin top-left"
    data_to_pixel_matrix: list[list[float]]
    pixel_to_data_matrix: list[list[float]]


class StyleMetadata(StrictModel):
    width_px: int
    height_px: int
    dpi: int
    line_width: float
    line_style: str
    line_color: str
    marker: str | None
    marker_size: float
    grid: bool
    title_visible: bool
    labels_visible: bool
    legend_visible: bool
    font_size: float
    tick_font_size: float
    background: str
    margins: dict[str, float]
    backend: Literal["matplotlib", "plotly"] = "matplotlib"


class CanonicalSample(StrictModel):
    schema_version: str = "0.1.0"
    sample_id: str
    latent_series_id: str
    rendering_id: str
    split: Literal["train", "validation", "test"]
    image_path: str
    challenge_group_id: str | None = None
    data: SeriesData
    axes: AxesMetadata
    image_geometry: ImageGeometry
    plot_area: PlotArea
    transform: TransformMetadata
    rendered_points: list[RenderedPoint]
    curve_grid: CurveGrid
    normalized_curve: NormalizedCurve
    pixel_curve: PixelCurve
    normalized_pixel_curve: NormalizedPixelCurve
    visual_resolution: VisualResolution
    style: StyleMetadata
    signal_structure: dict[str, Any] | None = None
