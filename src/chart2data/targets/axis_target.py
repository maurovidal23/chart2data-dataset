"""Axis calibration targets."""

from chart2data.schema import CanonicalSample


def build_axis_target(sample: CanonicalSample, axis: str = "y") -> dict[str, object]:
    metadata = sample.axes.y if axis == "y" else sample.axes.x
    return {
        f"{axis}_min": metadata.min,
        f"{axis}_max": metadata.max,
        "scale": metadata.scale,
        "ticks": metadata.ticks.copy(),
        "tick_labels": metadata.tick_labels.copy(),
    }
