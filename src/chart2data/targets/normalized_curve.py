"""Fixed-dimensional plot-normalized regression target."""

from chart2data.schema import CanonicalSample


def build_normalized_curve_target(sample: CanonicalSample) -> list[float]:
    return sample.normalized_curve.z.copy()
