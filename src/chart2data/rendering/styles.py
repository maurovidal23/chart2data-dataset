"""Independent visual style sampling."""

from __future__ import annotations

import numpy as np

from chart2data.config import RenderingConfig
from chart2data.schema import StyleMetadata


def sample_style(
    rng: np.random.Generator, config: RenderingConfig, *, backend: str = "matplotlib"
) -> StyleMetadata:
    width = int(rng.choice(config.widths_px))
    height = int(rng.choice(config.heights_px))
    dpi = int(rng.choice(config.dpis))
    marker_raw = str(rng.choice(["none", "o", "s", "^", "."]))
    left = float(rng.uniform(0.10, 0.18))
    right = float(rng.uniform(0.93, 0.98))
    bottom = float(rng.uniform(0.12, 0.20))
    top = float(rng.uniform(0.84, 0.95))
    return StyleMetadata(
        width_px=width,
        height_px=height,
        dpi=dpi,
        line_width=float(rng.uniform(1.0, 3.5)),
        line_style=str(rng.choice(["-", "--", "-.", ":"])),
        line_color=str(rng.choice(["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#111111"])),
        marker=None if marker_raw == "none" else marker_raw,
        marker_size=float(rng.uniform(2.5, 7.0)),
        grid=bool(rng.integers(0, 2)),
        title_visible=bool(rng.random() < 0.55),
        labels_visible=bool(rng.random() < 0.8),
        legend_visible=bool(rng.random() < 0.25),
        font_size=float(rng.uniform(9, 15)),
        tick_font_size=float(rng.uniform(7, 12)),
        background=str(rng.choice(["#ffffff", "#f5f5f5", "#fffdf5", "#eef3f8"])),
        margins={"left": left, "right": right, "bottom": bottom, "top": top},
        backend=backend,
    )
