"""Noise distributions."""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_noise(size: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    kind = str(rng.choice(["none", "gaussian", "student_t"], p=[0.15, 0.6, 0.25]))
    scale = float(rng.uniform(0.05, 2.5))
    if kind == "none":
        return np.zeros(size), {"type": kind, "std": 0.0}
    if kind == "gaussian":
        return rng.normal(0, scale, size), {"type": kind, "std": scale}
    degrees_freedom = float(rng.uniform(2.5, 8.0))
    return (
        rng.standard_t(degrees_freedom, size) * scale,
        {"type": kind, "scale": scale, "degrees_freedom": degrees_freedom},
    )
