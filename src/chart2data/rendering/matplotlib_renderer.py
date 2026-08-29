"""Matplotlib renderer that records actual post-draw geometry."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chart2data.config import RenderingConfig
from chart2data.rendering.axes import plan_axis
from chart2data.rendering.renderer import Renderer, RenderResult
from chart2data.rendering.styles import sample_style
from chart2data.rendering.transforms import matrices_from_matplotlib
from chart2data.schema import (
    AxesMetadata,
    AxisMetadata,
    ImageGeometry,
    PlotArea,
)


class MatplotlibRenderer(Renderer):
    def __init__(self, config: RenderingConfig) -> None:
        self.config = config

    def render(
        self, x: np.ndarray, y: np.ndarray, output_path: Path, rng: np.random.Generator
    ) -> RenderResult:
        style = sample_style(rng, self.config, backend="matplotlib")
        x_plan = plan_axis(x, rng, is_x=True)
        y_plan = plan_axis(y, rng)
        fig = plt.figure(
            figsize=(style.width_px / style.dpi, style.height_px / style.dpi),
            dpi=style.dpi,
            facecolor=style.background,
        )
        try:
            ax = fig.add_subplot(111, facecolor=style.background)
            fig.subplots_adjust(**style.margins)
            ax.plot(
                x,
                y,
                color=style.line_color,
                linewidth=style.line_width,
                linestyle=style.line_style,
                marker=style.marker,
                markersize=style.marker_size,
                label="Series",
            )
            ax.set_xlim(x_plan.minimum, x_plan.maximum)
            ax.set_ylim(y_plan.minimum, y_plan.maximum)
            ax.set_xticks(x_plan.ticks)
            ax.set_yticks(y_plan.ticks)
            ax.xaxis.set_major_formatter(x_plan.formatter)
            ax.yaxis.set_major_formatter(y_plan.formatter)
            ax.tick_params(labelsize=style.tick_font_size)
            if style.grid:
                ax.grid(True, alpha=0.35)
            else:
                ax.grid(False)
            if style.title_visible:
                ax.set_title("Synthetic time series", fontsize=style.font_size)
            if style.labels_visible:
                ax.set_xlabel("Time", fontsize=style.font_size)
                ax.set_ylabel("Value", fontsize=style.font_size)
            if style.legend_visible:
                ax.legend(fontsize=max(style.tick_font_size - 1, 5))

            fig.canvas.draw()
            width_px, height_px = fig.canvas.get_width_height()
            bbox = ax.get_window_extent()
            transform = matrices_from_matplotlib(ax.transData.get_affine().get_matrix(), height_px)
            x_labels = [label.get_text() for label in ax.get_xticklabels()]
            y_labels = [label.get_text() for label in ax.get_yticklabels()]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=style.dpi,
                facecolor=style.background,
                transparent=self.config.transparent,
            )
            axes = AxesMetadata(
                x=AxisMetadata(
                    min=float(ax.get_xlim()[0]),
                    max=float(ax.get_xlim()[1]),
                    ticks=[float(value) for value in ax.get_xticks()],
                    tick_labels=x_labels,
                    label=ax.get_xlabel() or None,
                    formatter=x_plan.formatter_name,
                ),
                y=AxisMetadata(
                    min=float(ax.get_ylim()[0]),
                    max=float(ax.get_ylim()[1]),
                    ticks=[float(value) for value in ax.get_yticks()],
                    tick_labels=y_labels,
                    label=ax.get_ylabel() or None,
                    formatter=y_plan.formatter_name,
                ),
            )
            return RenderResult(
                axes=axes,
                image_geometry=ImageGeometry(width_px=width_px, height_px=height_px, dpi=style.dpi),
                plot_area=PlotArea(
                    left_px=float(bbox.x0),
                    right_px=float(bbox.x1),
                    top_px=float(height_px - bbox.y1),
                    bottom_px=float(height_px - bbox.y0),
                ),
                transform=transform,
                style=style,
            )
        finally:
            plt.close(fig)
