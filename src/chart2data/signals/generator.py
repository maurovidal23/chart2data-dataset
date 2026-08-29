"""Composition of reproducible latent numerical series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chart2data.config import SignalConfig
from chart2data.signals.anomalies import add_spikes
from chart2data.signals.changepoints import add_changepoints
from chart2data.signals.noise import generate_noise
from chart2data.signals.seasonality import generate_seasonality
from chart2data.signals.trend import generate_trend


@dataclass(frozen=True)
class GeneratedSignal:
    x: np.ndarray
    y: np.ndarray
    structure: dict[str, Any]


def generate_signal(rng: np.random.Generator, config: SignalConfig) -> GeneratedSignal:
    """Generate a sorted, optionally irregular univariate series."""
    count = int(rng.integers(config.min_points, config.max_points + 1))
    if rng.random() < config.irregular_x_probability:
        steps = rng.uniform(0.4, 1.8, count)
        x = np.cumsum(steps)
        x -= x[0]
    else:
        spacing = float(rng.choice([0.25, 0.5, 1.0, 2.0]))
        x = np.arange(count, dtype=float) * spacing
    x_norm = (x - x[0]) / (x[-1] - x[0])
    trend, trend_meta = generate_trend(x_norm, rng)
    seasonal, seasonal_meta = generate_seasonality(x_norm, rng)
    noise, noise_meta = generate_noise(count, rng)
    y = trend + seasonal + noise
    y, changes = add_changepoints(y, x_norm, rng)
    y, anomalies = add_spikes(y, rng)
    structure = {
        "trend": trend_meta,
        "seasonality": seasonal_meta,
        "noise": noise_meta,
        "anomalies": anomalies,
        "change_points": changes,
    }
    return GeneratedSignal(x=x.astype(float), y=y.astype(float), structure=structure)
