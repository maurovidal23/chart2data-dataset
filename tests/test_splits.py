from __future__ import annotations

from chart2data.dataset.splits import assign_splits


def test_split_assignment_is_deterministic_and_series_level(small_config) -> None:
    first = assign_splits(100, small_config.split_ratios, 42)
    second = assign_splits(100, small_config.split_ratios, 42)
    assert first == second
    assert len(first) == 100
    assert set(first.values()) == {"train", "validation", "test"}
