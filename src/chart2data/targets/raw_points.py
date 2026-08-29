"""Raw data-space target views."""

from __future__ import annotations

from chart2data.schema import CanonicalSample


def build_raw_points_target(sample: CanonicalSample) -> dict[str, list[dict[str, float]]]:
    return {"points": [{"x": x, "y": y} for x, y in zip(sample.data.x, sample.data.y, strict=True)]}


def build_y_only_target(sample: CanonicalSample) -> dict[str, list[float]]:
    return {"y": sample.data.y.copy()}
