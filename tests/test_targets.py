from __future__ import annotations

import copy

from chart2data.targets import (
    build_axis_target,
    build_normalized_curve_target,
    build_pixel_curve_target,
    build_raw_points_target,
    build_vlm_json_target,
    build_y_only_target,
)


def test_target_builders_are_pure_and_complete(canonical_sample) -> None:
    before = copy.deepcopy(canonical_sample)
    assert len(build_raw_points_target(canonical_sample)["points"]) == len(canonical_sample.data.x)
    assert build_y_only_target(canonical_sample)["y"] == canonical_sample.data.y
    assert build_normalized_curve_target(canonical_sample) == canonical_sample.normalized_curve.z
    assert build_pixel_curve_target(canonical_sample) == canonical_sample.pixel_curve.y_px
    assert build_axis_target(canonical_sample)["ticks"] == canonical_sample.axes.y.ticks
    assert "points" in build_vlm_json_target(canonical_sample)
    assert canonical_sample == before
