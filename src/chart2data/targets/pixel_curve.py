"""Fixed-dimensional pixel curve target."""

from chart2data.schema import CanonicalSample


def build_pixel_curve_target(sample: CanonicalSample) -> list[float]:
    return sample.pixel_curve.y_px.copy()
