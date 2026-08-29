"""Sparse spike anomalies."""

from __future__ import annotations

from typing import Any

import numpy as np


def add_spikes(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = y.copy()
    anomalies: list[dict[str, Any]] = []
    count = int(rng.integers(0, min(4, max(1, len(y) // 12)) + 1))
    if count == 0:
        return result, anomalies
    indices = rng.choice(len(y), size=count, replace=False)
    base_scale = max(float(np.ptp(y)), 1.0)
    for raw_index in indices:
        index = int(raw_index)
        magnitude = float(rng.uniform(0.15, 0.65) * base_scale * rng.choice([-1, 1]))
        result[index] += magnitude
        anomalies.append({"type": "spike", "index": index, "magnitude": magnitude})
    return result, anomalies
