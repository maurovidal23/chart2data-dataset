from __future__ import annotations

import hashlib
import json
import sys

import pytest

from chart2data.config import DatasetConfig, RenderingConfig
from chart2data.random import make_rng
from chart2data.rendering import MatplotlibRenderer, get_renderer, resolve_backend


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rendering_config(backend: str) -> RenderingConfig:
    return RenderingConfig(widths_px=[400], heights_px=[300], dpis=[100], backend=backend)


def test_default_backend_is_matplotlib_and_factory_is_lazy() -> None:
    config = RenderingConfig()
    assert config.backend == "matplotlib"
    assert isinstance(get_renderer(config), MatplotlibRenderer)


def test_matplotlib_factory_does_not_import_optional_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "kaleido", None)
    config = RenderingConfig()
    assert isinstance(get_renderer(config), MatplotlibRenderer)


def test_resolve_backend_is_deterministic_per_stream() -> None:
    config = _rendering_config("auto")
    first_seed = make_rng(42, "series_00000000", "render_00", "backend")
    second_seed = make_rng(42, "series_00000000", "render_00", "backend")
    assert resolve_backend(config, first_seed) == resolve_backend(config, second_seed)
    assert resolve_backend(config, first_seed) in ("matplotlib", "plotly")


def test_resolve_backend_only_selects_known_backends() -> None:
    config = _rendering_config("auto")
    rng = make_rng(7, "sweep", "render_00", "backend")
    choices = {resolve_backend(config, rng) for _ in range(200)}
    assert choices <= {"matplotlib", "plotly"}


def test_default_config_backward_compatible_hash() -> None:
    config = DatasetConfig()
    payload = json.loads(json.dumps(config.model_dump(mode="json"), sort_keys=True))
    rendering = payload["rendering"]
    assert rendering.pop("backend") == "matplotlib"
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert config.stable_hash() == _hash(payload_text.encode())


def test_explicit_matplotlib_config_collides_with_implicit_default_hash() -> None:
    implicit = DatasetConfig()
    explicit = DatasetConfig(rendering=RenderingConfig(backend="matplotlib"))
    assert implicit.stable_hash() == explicit.stable_hash()


def test_unknown_backend_is_rejected_by_config() -> None:
    with pytest.raises(ValueError):
        DatasetConfig(rendering=RenderingConfig(backend="svg"))


def test_plotly_factory_raises_actionable_error_when_optional_deps_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "kaleido", None)
    config = _rendering_config("plotly")
    with pytest.raises(RuntimeError) as excinfo:
        get_renderer(config)
    message = str(excinfo.value)
    assert "pip install" in message
    assert "plotly" in message
    assert "never silently falls back" in message


def test_plotly_factory_raises_when_kaleido_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "kaleido", None)
    with pytest.raises(RuntimeError) as excinfo:
        get_renderer(_rendering_config("plotly"))
    assert "plotly" in str(excinfo.value)