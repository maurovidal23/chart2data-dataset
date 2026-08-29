"""Stable, order-independent random streams."""

from __future__ import annotations

import hashlib

import numpy as np


def _stable_uint32(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


def make_rng(global_seed: int, *keys: str | int) -> np.random.Generator:
    """Create a deterministic RNG from a global seed and semantic keys."""
    entropy = [global_seed & 0xFFFFFFFF, *(_stable_uint32(str(key)) for key in keys)]
    return np.random.default_rng(np.random.SeedSequence(entropy))
