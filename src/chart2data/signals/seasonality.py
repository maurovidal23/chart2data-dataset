"""One or more sinusoidal components."""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_seasonality(
    x_norm: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.zeros_like(x_norm)
    metadata: list[dict[str, Any]] = []
    count = int(rng.integers(0, 4))
    for _ in range(count):
        cycles = float(rng.uniform(0.5, 8.0))
        amplitude = float(rng.uniform(0.5, 12.0))
        phase = float(rng.uniform(0, 2 * np.pi))
        result += amplitude * np.sin(2 * np.pi * cycles * x_norm + phase)
        metadata.append({"period": 1.0 / cycles, "amplitude": amplitude, "phase": phase})
    return result, metadata
