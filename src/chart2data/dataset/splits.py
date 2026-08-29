"""Leakage-safe deterministic latent-series splitting."""

from __future__ import annotations

import numpy as np

from chart2data.config import SplitRatios
from chart2data.random import make_rng


def assign_splits(num_latent_series: int, ratios: SplitRatios, global_seed: int) -> dict[str, str]:
    if num_latent_series < 1:
        raise ValueError("num_latent_series must be positive")
    identifiers = [f"series_{index:08d}" for index in range(num_latent_series)]
    order = make_rng(global_seed, "split-assignment").permutation(num_latent_series)
    train_count = int(np.floor(num_latent_series * ratios.train))
    validation_count = int(np.floor(num_latent_series * ratios.validation))
    assignments: dict[str, str] = {}
    for position, raw_index in enumerate(order):
        if position < train_count:
            split = "train"
        elif position < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        assignments[identifiers[int(raw_index)]] = split
    return assignments


def split_counts(assignments: dict[str, str]) -> dict[str, int]:
    return {
        name: sum(value == name for value in assignments.values())
        for name in ("train", "validation", "test")
    }
