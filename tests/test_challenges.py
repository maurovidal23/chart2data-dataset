from __future__ import annotations

from pathlib import Path

import numpy as np

from chart2data.config import AxisChallengeConfig
from chart2data.dataset.builder import _generate_series


def test_axis_challenge_pair_has_equal_geometry_and_scaled_truth(
    tmp_path: Path, small_config
) -> None:
    config = small_config.model_copy(
        update={
            "axis_challenges": AxisChallengeConfig(
                enabled=True, fraction=1.0, scale_factors=[1000.0]
            )
        }
    )
    payload = config.model_dump(mode="json")
    first = _generate_series(0, "train", payload, str(tmp_path))[0]
    second = _generate_series(1, "train", payload, str(tmp_path))[0]
    assert first["challenge_group_id"] == second["challenge_group_id"]
    np.testing.assert_allclose(second["data"]["y"], np.asarray(first["data"]["y"]) * 1000)
    np.testing.assert_allclose(first["normalized_curve"]["z"], second["normalized_curve"]["z"])
    assert first["plot_area"] == second["plot_area"]
    assert first["axes"]["y"]["tick_labels"] != second["axes"]["y"]["tick_labels"]
