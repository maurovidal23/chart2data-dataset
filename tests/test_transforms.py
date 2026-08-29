from __future__ import annotations

import numpy as np

from chart2data.rendering.transforms import data_to_pixel, pixel_to_data


def test_coordinate_round_trip(canonical_sample) -> None:
    points = np.column_stack([canonical_sample.data.x, canonical_sample.data.y])
    recovered = pixel_to_data(
        data_to_pixel(points, canonical_sample.transform), canonical_sample.transform
    )
    np.testing.assert_allclose(recovered, points, rtol=1e-12, atol=1e-12)
