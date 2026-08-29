"""A conventional JSON target; one view among many, never canonical truth."""

from chart2data.schema import CanonicalSample
from chart2data.targets.raw_points import build_raw_points_target


def build_vlm_json_target(sample: CanonicalSample) -> dict[str, object]:
    return {
        "axis": {"y_min": sample.axes.y.min, "y_max": sample.axes.y.max},
        **build_raw_points_target(sample),
    }
