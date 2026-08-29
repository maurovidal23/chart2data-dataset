"""Renderer interface and result contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chart2data.schema import (
    AxesMetadata,
    ImageGeometry,
    PlotArea,
    StyleMetadata,
    TransformMetadata,
)


@dataclass(frozen=True)
class RenderResult:
    axes: AxesMetadata
    image_geometry: ImageGeometry
    plot_area: PlotArea
    transform: TransformMetadata
    style: StyleMetadata


class Renderer(ABC):
    @abstractmethod
    def render(
        self, x: np.ndarray, y: np.ndarray, output_path: Path, rng: np.random.Generator
    ) -> RenderResult:
        """Render a chart and return renderer-observed canonical metadata."""
