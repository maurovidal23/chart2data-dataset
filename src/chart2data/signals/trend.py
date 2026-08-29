"""Trend component generation."""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_trend(
    x_norm: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    kind = str(rng.choice(["constant", "linear", "quadratic"], p=[0.2, 0.5, 0.3]))
    intercept = float(rng.uniform(-20, 20))
    if kind == "constant":
        return np.full_like(x_norm, intercept), {"type": kind, "parameters": {"level": intercept}}
    slope = float(rng.uniform(-35, 35))
    values = intercept + slope * x_norm
    parameters: dict[str, float] = {"intercept": intercept, "slope": slope}
    if kind == "quadratic":
        curvature = float(rng.uniform(-35, 35))
        values = values + curvature * (x_norm - 0.5) ** 2
        parameters["curvature"] = curvature
    return values, {"type": kind, "parameters": parameters}
