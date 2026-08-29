from __future__ import annotations

import numpy as np

from chart2data.metadata.geometry import denormalize_y, normalize_y, resample_curve
from chart2data.metadata.serialization import read_jsonl, read_parquet, write_jsonl, write_parquet


def test_fixed_grid_interpolation_for_known_linear_function() -> None:
    x = np.array([0.0, 1.0, 3.0, 4.0])
    y = 3.5 * x - 2.0
    grid = resample_curve(x, y, 17)
    np.testing.assert_allclose(grid.y, 3.5 * np.asarray(grid.x) - 2.0)


def test_normalization_round_trip(canonical_sample) -> None:
    y = np.asarray(canonical_sample.curve_grid.y)
    z = normalize_y(y, canonical_sample.axes.y)
    np.testing.assert_allclose(denormalize_y(z, canonical_sample.axes.y), y)
    assert np.all((z >= 0) & (z <= 1))


def test_serialization_preserves_canonical_values(tmp_path, canonical_sample) -> None:
    jsonl = tmp_path / "sample.jsonl"
    parquet = tmp_path / "sample.parquet"
    write_jsonl([canonical_sample], jsonl)
    write_parquet([canonical_sample], parquet)
    from_json = read_jsonl(jsonl)[0]
    from_parquet = read_parquet(parquet)[0]
    assert from_json == canonical_sample
    assert from_parquet == canonical_sample
    assert from_parquet.data.y == canonical_sample.data.y
