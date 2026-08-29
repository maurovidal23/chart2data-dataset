"""Plotly/Kaleido renderer that records analytic post-layout geometry.

Plotly is an optional rendering backend. Kaleido handles static image export and
must be importable (Kaleido 1.x additionally requires a system Chrome runtime;
Kaleido 0.2.x provisions a self-contained headless browser). Imports and rendering
fail loudly with actionable messages rather than silently falling back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chart2data.config import RenderingConfig
from chart2data.rendering.axes import plan_axis
from chart2data.rendering.renderer import Renderer, RenderResult
from chart2data.rendering.styles import sample_style
from chart2data.rendering.transforms import matrices_from_affine
from chart2data.schema import AxesMetadata, AxisMetadata, ImageGeometry, PlotArea

PLOTLY_IMPORT_ERROR = (
    "The 'plotly' rendering backend was requested but its dependencies are not installed. "
    "Install the optional extra, e.g. `pip install -e '.[plotly]'`. The backend never "
    "silently falls back to matplotlib."
)

KALEIDO_RENDER_ERROR = (
    "Static rendering through Kaleido failed. On Kaleido 1.x run `kaleido_get_chrome` "
    "(requires network access and authorization to download Chrome). Kaleido 0.2.x "
    "downloads a self-contained headless browser on first use. The backend never "
    "silently falls back to matplotlib."
)

_LINE_STYLES = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}
_MARKER_SYMBOLS = {"o": "circle", "s": "square", "^": "triangle-up", ".": "circle"}


def _margin_px(fraction: float, reference_px: int) -> int:
    return max(0, round(fraction * reference_px))


def _hex_color(value: str) -> str:
    return value if value.startswith("#") else f"#{value}"


class PlotlyRenderer(Renderer):
    def __init__(self, config: RenderingConfig) -> None:
        self.config = config

    def render(
        self, x: np.ndarray, y: np.ndarray, output_path: Path, rng: np.random.Generator
    ) -> RenderResult:
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError as exc:
            raise RuntimeError(PLOTLY_IMPORT_ERROR) from exc

        style = sample_style(rng, self.config, backend="plotly")
        x_plan = plan_axis(x, rng, is_x=True)
        y_plan = plan_axis(y, rng)
        width = int(style.width_px)
        height = int(style.height_px)
        margin_l = _margin_px(style.margins["left"], width)
        margin_r = _margin_px(1.0 - style.margins["right"], width)
        margin_t = _margin_px(1.0 - style.margins["top"], height)
        margin_b = _margin_px(style.margins["bottom"], height)

        x_labels = [x_plan.formatter(value, 0) for value in x_plan.ticks]
        y_labels = [y_plan.formatter(value, 0) for value in y_plan.ticks]

        fig = go.Figure()
        trace = go.Scatter(
            x=x.tolist(),
            y=y.tolist(),
            mode="lines" if style.marker is None else "lines+markers",
            line={
                "color": _hex_color(style.line_color),
                "width": style.line_width,
                "dash": _LINE_STYLES.get(style.line_style, "solid"),
            },
            marker=None
            if style.marker is None
            else {
                "symbol": _MARKER_SYMBOLS.get(style.marker, "circle"),
                "size": style.marker_size,
                "color": _hex_color(style.line_color),
            },
            name="Series",
            hoverinfo="skip",
        )
        fig.add_trace(trace)
        fig.update_layout(
            width=width,
            height=height,
            margin={"l": margin_l, "r": margin_r, "t": margin_t, "b": margin_b, "pad": 0},
            paper_bgcolor=style.background,
            plot_bgcolor=style.background,
            font={"family": "sans-serif", "size": style.tick_font_size},
            showlegend=style.legend_visible,
            legend={
                "x": 0.98,
                "y": 0.98,
                "xanchor": "right",
                "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.85)",
                "bordercolor": "rgba(0, 0, 0, 0.15)",
                "font": {"size": max(style.tick_font_size - 1, 5)},
            },
            title={"text": "Synthetic time series", "font": {"size": style.font_size}}
            if style.title_visible
            else None,
            xaxis={
                "range": [x_plan.minimum, x_plan.maximum],
                "tickmode": "array",
                "tickvals": x_plan.ticks.tolist(),
                "ticktext": x_labels,
                "tickfont": {"size": style.tick_font_size},
                "title": {"text": "Time", "font": {"size": style.font_size}, "standoff": 2}
                if style.labels_visible
                else None,
                "automargin": False,
                "fixedrange": True,
                "showgrid": style.grid,
                "showline": False,
                "zeroline": False,
            },
            yaxis={
                "range": [y_plan.minimum, y_plan.maximum],
                "tickmode": "array",
                "tickvals": y_plan.ticks.tolist(),
                "ticktext": y_labels,
                "tickfont": {"size": style.tick_font_size},
                "title": {"text": "Value", "font": {"size": style.font_size}, "standoff": 2}
                if style.labels_visible
                else None,
                "automargin": False,
                "fixedrange": True,
                "showgrid": style.grid,
                "showline": False,
                "zeroline": False,
            },
        )

        try:
            image_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=1)
        except Exception as exc:
            raise RuntimeError(KALEIDO_RENDER_ERROR) from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        plot_left = float(margin_l)
        plot_right = float(width - margin_r)
        plot_top = float(margin_t)
        plot_bottom = float(height - margin_b)
        plot_width_px = plot_right - plot_left
        plot_height_px = plot_bottom - plot_top
        x_span = x_plan.maximum - x_plan.minimum
        y_span = y_plan.maximum - y_plan.minimum
        px_per_unit_x = plot_width_px / x_span
        px_per_unit_y = plot_height_px / y_span
        offset_x = plot_left - x_plan.minimum * px_per_unit_x
        offset_y = plot_top + y_plan.maximum * px_per_unit_y
        transform = matrices_from_affine(px_per_unit_x, offset_x, -px_per_unit_y, offset_y)

        axes = AxesMetadata(
            x=AxisMetadata(
                min=float(x_plan.minimum),
                max=float(x_plan.maximum),
                ticks=[float(value) for value in x_plan.ticks],
                tick_labels=x_labels,
                label="Time" if style.labels_visible else None,
                formatter=x_plan.formatter_name,
            ),
            y=AxisMetadata(
                min=float(y_plan.minimum),
                max=float(y_plan.maximum),
                ticks=[float(value) for value in y_plan.ticks],
                tick_labels=y_labels,
                label="Value" if style.labels_visible else None,
                formatter=y_plan.formatter_name,
            ),
        )
        return RenderResult(
            axes=axes,
            image_geometry=ImageGeometry(width_px=width, height_px=height, dpi=style.dpi),
            plot_area=PlotArea(
                left_px=plot_left,
                right_px=plot_right,
                top_px=plot_top,
                bottom_px=plot_bottom,
            ),
            transform=transform,
            style=style,
        )