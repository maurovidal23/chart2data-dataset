from __future__ import annotations

from pathlib import Path

import pytest

from chart2data.config import DatasetConfig, RenderingConfig, SignalConfig
from chart2data.metadata.canonical import build_canonical_sample
from chart2data.random import make_rng
from chart2data.rendering.matplotlib_renderer import MatplotlibRenderer
from chart2data.signals.generator import generate_signal


@pytest.fixture
def small_config() -> DatasetConfig:
    return DatasetConfig(
        num_latent_series=10,
        renderings_per_series=2,
        curve_grid_size=24,
        signal=SignalConfig(min_points=12, max_points=12, irregular_x_probability=0.0),
        rendering=RenderingConfig(widths_px=[400], heights_px=[300], dpis=[100]),
    )


@pytest.fixture
def canonical_sample(tmp_path: Path, small_config: DatasetConfig):
    latent_id = "series_00000000"
    signal = generate_signal(make_rng(42, latent_id, "latent"), small_config.signal)
    rendered = MatplotlibRenderer(small_config.rendering).render(
        signal.x,
        signal.y,
        tmp_path / "sample.png",
        make_rng(42, latent_id, "render_00"),
    )
    return build_canonical_sample(
        sample_id="00000000",
        latent_series_id=latent_id,
        rendering_id="render_00",
        split="train",
        image_path="sample.png",
        x=signal.x,
        y=signal.y,
        rendered=rendered,
        curve_grid_size=small_config.curve_grid_size,
        signal_structure=signal.structure,
    )
