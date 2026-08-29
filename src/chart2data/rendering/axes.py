"""Randomized linear axis calibration and formatting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
class AxisPlan:
    minimum: float
    maximum: float
    ticks: np.ndarray
    formatter_name: str
    formatter: FuncFormatter


def _format_function(name: str, decimals: int):
    if name == "scientific":
        return lambda value, _position: f"{value:.{decimals}e}"
    if name == "comma":
        return lambda value, _position: f"{value:,.{decimals}f}"
    return lambda value, _position: f"{value:.{decimals}f}"


def plan_axis(values: np.ndarray, rng: np.random.Generator, *, is_x: bool = False) -> AxisPlan:
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    span = data_max - data_min
    if span <= 0:
        span = max(abs(data_min), 1.0)

    mode = str(rng.choice(["tight", "padded", "wide", "include_zero"], p=[0.2, 0.4, 0.25, 0.15]))
    if is_x:
        mode = str(rng.choice(["tight", "padded"], p=[0.45, 0.55]))
    if mode == "tight":
        lower_pad, upper_pad = rng.uniform(0.01, 0.08, 2) * span
    elif mode == "padded":
        lower_pad, upper_pad = rng.uniform(0.08, 0.4, 2) * span
    elif mode == "wide":
        lower_pad, upper_pad = rng.uniform(0.5, 3.0, 2) * span
    else:
        lower_pad, upper_pad = rng.uniform(0.08, 0.35, 2) * span
    minimum = data_min - float(lower_pad)
    maximum = data_max + float(upper_pad)
    if mode == "include_zero":
        minimum = min(0.0, minimum)
        maximum = max(0.0, maximum)
    if maximum <= minimum:
        maximum = minimum + 1.0

    tick_count = int(rng.integers(4, 9))
    ticks = np.linspace(minimum, maximum, tick_count)
    displayed_span = maximum - minimum
    if displayed_span >= 1e5 or displayed_span < 1e-3:
        formatter_name = "scientific"
        decimals = int(rng.integers(1, 4))
    else:
        formatter_name = str(rng.choice(["decimal", "comma"], p=[0.8, 0.2]))
        rough_step = displayed_span / max(tick_count - 1, 1)
        natural_decimals = max(0, int(np.ceil(-np.log10(max(rough_step, 1e-12)))) + 1)
        decimals = int(min(4, natural_decimals + int(rng.integers(0, 2))))
    function = _format_function(formatter_name, decimals)
    return AxisPlan(
        minimum=minimum,
        maximum=maximum,
        ticks=ticks,
        formatter_name=f"{formatter_name}:{decimals}",
        formatter=FuncFormatter(function),
    )


def axis_scale_challenge(y: np.ndarray, factor: float) -> tuple[np.ndarray, str]:
    """Create a numerically scaled counterpart with identical normalized shape."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    return y * factor, f"axis-scale-x{factor:g}"
