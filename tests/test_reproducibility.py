from __future__ import annotations

from pathlib import Path

from chart2data.metadata.canonical import build_canonical_sample
from chart2data.random import make_rng
from chart2data.rendering.matplotlib_renderer import MatplotlibRenderer
from chart2data.signals.generator import generate_signal


def _make(path: Path, config):
    latent = "series_00000007"
    signal = generate_signal(make_rng(123, latent, "latent"), config.signal)
    rendered = MatplotlibRenderer(config.rendering).render(
        signal.x, signal.y, path, make_rng(123, latent, "render_01")
    )
    return build_canonical_sample(
        sample_id="00000015",
        latent_series_id=latent,
        rendering_id="render_01",
        split="test",
        image_path="image.png",
        x=signal.x,
        y=signal.y,
        rendered=rendered,
        curve_grid_size=config.curve_grid_size,
        signal_structure=signal.structure,
    )


def test_exact_metadata_and_png_reproducibility(tmp_path, small_config) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = _make(first_path, small_config)
    second = _make(second_path, small_config)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
