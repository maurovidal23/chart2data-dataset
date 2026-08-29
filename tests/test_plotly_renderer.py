from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chart2data.config import AxisChallengeConfig, DatasetConfig
from chart2data.dataset.builder import _generate_series, build_dataset
from chart2data.metadata.serialization import read_jsonl
from chart2data.random import make_rng
from chart2data.rendering import get_renderer
from chart2data.rendering.transforms import data_to_pixel
from chart2data.signals.generator import generate_signal
from chart2data.validation.checks import validate_sample


def _plotly_installed() -> bool:
    try:
        import kaleido  # noqa: F401
        import plotly  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _plotly_installed(), reason="plotly/kaleido optional dependencies not installed"
)


def _plotly_config(small_config) -> DatasetConfig:
    return small_config.model_copy(
        update={"rendering": small_config.rendering.model_copy(update={"backend": "plotly"})}
    )


def _render_plotly_sample(tmp_path: Path, config: DatasetConfig, seed: int = 42):
    latent = "series_00000000"
    signal = generate_signal(make_rng(seed, latent, "latent"), config.signal)
    rendered = get_renderer(config.rendering, "plotly").render(
        signal.x, signal.y, tmp_path / "sample.png", make_rng(seed, latent, "render_00")
    )
    return rendered, signal


def test_plotly_contract_geometry_and_validators(tmp_path, small_config) -> None:
    rendered, signal = _render_plotly_sample(tmp_path, _plotly_config(small_config))
    image = Image.open(tmp_path / "sample.png")
    assert image.size == (rendered.image_geometry.width_px, rendered.image_geometry.height_px)
    assert rendered.style.backend == "plotly"
    area = rendered.plot_area
    assert 0 < area.left_px < area.right_px < image.width
    assert 0 < area.top_px < area.bottom_px < image.height
    assert len(rendered.axes.x.ticks) == len(rendered.axes.x.tick_labels) >= 2
    assert len(rendered.axes.y.ticks) == len(rendered.axes.y.tick_labels) >= 2
    for label in rendered.axes.x.tick_labels + rendered.axes.y.tick_labels:
        float(label.replace(",", ""))  # committed truth must be parseable numbers
    assert np.linalg.det(np.asarray(rendered.transform.data_to_pixel_matrix)) != 0
    pixels = data_to_pixel(np.column_stack([signal.x, signal.y]), rendered.transform)
    assert np.all((pixels[:, 0] >= area.left_px) & (pixels[:, 0] <= area.right_px))
    assert np.all((pixels[:, 1] >= area.top_px) & (pixels[:, 1] <= area.bottom_px))


def test_plotly_renderer_deterministic_metadata_and_bytes(tmp_path, small_config) -> None:
    config = _plotly_config(small_config)
    latent = "series_00000007"
    signal = generate_signal(make_rng(123, latent, "latent"), config.signal)
    left = get_renderer(config.rendering, "plotly").render(
        signal.x, signal.y, tmp_path / "a.png", make_rng(123, latent, "render_01")
    )
    right = get_renderer(config.rendering, "plotly").render(
        signal.x, signal.y, tmp_path / "b.png", make_rng(123, latent, "render_01")
    )
    assert left == right
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()


def test_plotly_pixel_probe_anchors_transform(tmp_path, small_config) -> None:
    """Persisted transforms must match real rendered pixels, not just be self-consistent."""
    config = _plotly_config(small_config)
    worst = 0.0
    total = 0
    for seed in (11, 22, 33):
        rendered, signal = _render_plotly_sample(tmp_path, config, seed=seed)
        img = np.asarray(Image.open(tmp_path / "sample.png").convert("RGB")).astype(int)
        line_rgb = np.array(
            [int(rendered.style.line_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
        )
        dist = np.sqrt(((img - line_rgb) ** 2).sum(axis=-1))
        predicted = data_to_pixel(np.column_stack([signal.x, signal.y]), rendered.transform)
        for px, py in predicted:
            px, py = round(px), round(py)
            box = dist[max(0, py - 4) : py + 5, max(0, px - 4) : px + 5]
            ys, xs = np.where(box < 200)
            assert len(xs) > 0, f"no line pixel near predicted point ({px}, {py})"
            cx = max(0, px - 4) + float(xs.mean())
            cy = max(0, py - 4) + float(ys.mean())
            worst = max(worst, max(abs(cx - px), abs(cy - py)))
            total += 1
    assert total > 0
    assert worst <= 4.0, f"rendered line diverges from analytic transform by {worst}px"


def test_plotly_auto_matches_explicit_backend_samples(tmp_path, small_config) -> None:
    auto = small_config.model_copy(
        update={"rendering": small_config.rendering.model_copy(update={"backend": "auto"})}
    )
    payload = auto.model_dump(mode="json")
    seen = set()
    for index in range(6):
        auto_record = _generate_series(index, "train", payload, str(tmp_path / "auto"))[0]
        chosen = auto_record["style"]["backend"]
        seen.add(chosen)
        explicit = auto.model_copy(
            update={"rendering": auto.rendering.model_copy(update={"backend": chosen})}
        )
        explicit_record = _generate_series(
            index, "train", explicit.model_dump(mode="json"), str(tmp_path / "explicit")
        )[0]
        assert auto_record == explicit_record
    assert seen == {"matplotlib", "plotly"}


def test_plotly_challenge_pair_shares_geometry_and_rendering(tmp_path, small_config) -> None:
    config = small_config.model_copy(
        update={
            "rendering": small_config.rendering.model_copy(update={"backend": "plotly"}),
            "axis_challenges": AxisChallengeConfig(
                enabled=True, fraction=1.0, scale_factors=[1000.0]
            ),
        }
    )
    payload = config.model_dump(mode="json")
    first = _generate_series(0, "train", payload, str(tmp_path))[0]
    second = _generate_series(1, "train", payload, str(tmp_path))[0]
    assert first["challenge_group_id"] == second["challenge_group_id"]
    assert first["style"]["backend"] == second["style"]["backend"] == "plotly"
    np.testing.assert_allclose(second["data"]["y"], np.asarray(first["data"]["y"]) * 1000)
    np.testing.assert_allclose(
        first["normalized_curve"]["z"], second["normalized_curve"]["z"]
    )
    np.testing.assert_allclose(
        first["pixel_curve"]["x_px"], second["pixel_curve"]["x_px"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        first["pixel_curve"]["y_px"], second["pixel_curve"]["y_px"], rtol=1e-12, atol=1e-12
    )
    assert first["plot_area"] == second["plot_area"]
    assert first["axes"]["y"]["tick_labels"] != second["axes"]["y"]["tick_labels"]


def test_plotly_build_dataset_end_to_end(tmp_path, small_config) -> None:
    config = small_config.model_copy(
        update={
            "num_latent_series": 2,
            "renderings_per_series": 1,
            "rendering": small_config.rendering.model_copy(update={"backend": "plotly"}),
        }
    )
    manifest = build_dataset(config, tmp_path / "dataset")
    assert manifest["num_images"] == 2
    assert manifest["renderer_environments"]["plotly"] != "not installed"
    samples = [
        sample
        for split in ("train", "validation", "test")
        for sample in read_jsonl(tmp_path / "dataset" / "jsonl" / f"{split}.jsonl")
    ]
    assert len(samples) == 2
    for sample in samples:
        assert sample.style.backend == "plotly"
        assert validate_sample(sample) == []


def test_multiprocess_build_is_deterministic_across_workers(tmp_path, small_config) -> None:
    config = small_config.model_copy(
        update={
            "num_latent_series": 4,
            "renderings_per_series": 2,
            "rendering": small_config.rendering.model_copy(update={"backend": "auto"}),
        }
    )
    serial = config.model_copy(update={"num_workers": 1})
    parallel = config.model_copy(update={"num_workers": 4})
    build_dataset(serial, tmp_path / "serial")
    build_dataset(parallel, tmp_path / "parallel")

    def read_records(root: Path) -> dict[str, dict]:
        records = {}
        for split in ("train", "validation", "test"):
            path = root / "jsonl" / f"{split}.jsonl"
            for line in path.read_text().splitlines():
                record = json.loads(line)
                records[record["sample_id"]] = record
        return records

    serial_records = read_records(tmp_path / "serial")
    parallel_records = read_records(tmp_path / "parallel")
    assert serial_records == parallel_records
    for sample_id, record in serial_records.items():
        serial_image = (tmp_path / "serial" / record["image_path"]).read_bytes()
        parallel_path = tmp_path / "parallel" / parallel_records[sample_id]["image_path"]
        assert serial_image == parallel_path.read_bytes()