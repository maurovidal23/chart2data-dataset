from __future__ import annotations

import numpy as np
from PIL import Image

from chart2data.metadata.canonical import build_canonical_sample
from chart2data.random import make_rng
from chart2data.rendering.matplotlib_renderer import MatplotlibRenderer
from chart2data.signals.generator import generate_signal


def test_pixel_geometry_matches_png(canonical_sample, tmp_path) -> None:
    image = Image.open(tmp_path / canonical_sample.image_path)
    assert image.size == (
        canonical_sample.image_geometry.width_px,
        canonical_sample.image_geometry.height_px,
    )
    area = canonical_sample.plot_area
    assert 0 < area.left_px < area.right_px < image.width
    assert 0 < area.top_px < area.bottom_px < image.height
    x_px = np.asarray(canonical_sample.pixel_curve.x_px)
    y_px = np.asarray(canonical_sample.pixel_curve.y_px)
    assert np.all((x_px >= area.left_px) & (x_px <= area.right_px))
    assert np.all((y_px >= area.top_px) & (y_px <= area.bottom_px))


def test_axis_variation_preserves_latent_truth(tmp_path, small_config) -> None:
    latent = "series_00000003"
    signal = generate_signal(make_rng(42, latent, "latent"), small_config.signal)
    renderer = MatplotlibRenderer(small_config.rendering)
    samples = []
    for index in range(2):
        rendering_id = f"render_{index:02d}"
        rendered = renderer.render(
            signal.x,
            signal.y,
            tmp_path / f"{index}.png",
            make_rng(42, latent, rendering_id),
        )
        samples.append(
            build_canonical_sample(
                sample_id=f"{index:08d}",
                latent_series_id=latent,
                rendering_id=rendering_id,
                split="train",
                image_path=f"{index}.png",
                x=signal.x,
                y=signal.y,
                rendered=rendered,
                curve_grid_size=small_config.curve_grid_size,
                signal_structure=None,
            )
        )
    assert samples[0].data == samples[1].data
    assert (samples[0].axes.y.min, samples[0].axes.y.max) != (
        samples[1].axes.y.min,
        samples[1].axes.y.max,
    )
