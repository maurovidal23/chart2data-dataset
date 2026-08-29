"""Chart rendering and coordinate transforms."""

from __future__ import annotations

import numpy as np

from chart2data.config import RenderingConfig
from chart2data.rendering.matplotlib_renderer import MatplotlibRenderer
from chart2data.rendering.plotly_renderer import (
    KALEIDO_RENDER_ERROR,
    PLOTLY_IMPORT_ERROR,
    PlotlyRenderer,
)
from chart2data.rendering.renderer import Renderer, RenderResult

__all__ = [
    "KALEIDO_RENDER_ERROR",
    "PLOTLY_IMPORT_ERROR",
    "MatplotlibRenderer",
    "PlotlyRenderer",
    "RenderResult",
    "Renderer",
    "get_renderer",
    "resolve_backend",
]

_BACKENDS = ("matplotlib", "plotly")


def resolve_backend(config: RenderingConfig, rng: np.random.Generator) -> str:
    """Determine the concrete backend for a single rendering.

    ``auto`` is sampled deterministically from ``rng``. The sampling stream is the
    caller's responsibility (keyed independently of the style/axis stream), so adding
    or reordering backends never perturbs an existing rendering stream, and challenge
    pairs that share a rendering key select the same backend.
    """
    if config.backend != "auto":
        return config.backend
    return str(rng.choice(_BACKENDS))


def get_renderer(config: RenderingConfig, backend: str | None = None) -> Renderer:
    """Instantiate the renderer for a concrete backend, failing loudly if optional
    dependencies are missing."""
    chosen = config.backend if backend is None else backend
    if chosen == "matplotlib":
        return MatplotlibRenderer(config)
    if chosen == "plotly":
        try:
            import kaleido  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(PLOTLY_IMPORT_ERROR) from exc
        return PlotlyRenderer(config)
    raise ValueError(f"unknown rendering backend {chosen!r}")