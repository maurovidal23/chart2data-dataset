from __future__ import annotations

import numpy as np

from chart2data.config import SignalConfig
from chart2data.random import make_rng
from chart2data.signals.generator import generate_signal


def test_signal_contains_exact_aligned_finite_values() -> None:
    signal = generate_signal(
        make_rng(91, "series_4", "latent"),
        SignalConfig(min_points=20, max_points=20, irregular_x_probability=1.0),
    )
    assert len(signal.x) == len(signal.y) == 20
    assert np.all(np.diff(signal.x) > 0)
    assert np.all(np.isfinite(signal.y))
    assert {
        "trend",
        "seasonality",
        "noise",
        "anomalies",
        "change_points",
    } <= signal.structure.keys()
