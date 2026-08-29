"""Validated configuration and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitRatios(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def validate_sum(self) -> SplitRatios:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        if min(self.train, self.validation, self.test) < 0:
            raise ValueError("split ratios must be non-negative")
        return self


class SignalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_points: int = Field(32, ge=2)
    max_points: int = Field(128, ge=2)
    irregular_x_probability: float = Field(0.25, ge=0, le=1)

    @model_validator(mode="after")
    def validate_points(self) -> SignalConfig:
        if self.max_points < self.min_points:
            raise ValueError("max_points must be >= min_points")
        return self


class RenderingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    widths_px: list[int] = Field(default_factory=lambda: [640, 800, 1024])
    heights_px: list[int] = Field(default_factory=lambda: [480, 600, 768])
    dpis: list[int] = Field(default_factory=lambda: [80, 100, 120])
    transparent: bool = False
    backend: Literal["matplotlib", "plotly", "auto"] = "matplotlib"


class AxisChallengeConfig(BaseModel):
    """Settings for paired equal-shape, different-scale latent series."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    fraction: float = Field(0.0, ge=0, le=1)
    scale_factors: list[float] = Field(default_factory=lambda: [100.0, 1000.0, 1_000_000.0])

    @model_validator(mode="after")
    def validate_factors(self) -> AxisChallengeConfig:
        if not self.scale_factors or any(
            factor <= 0 or factor == 1 for factor in self.scale_factors
        ):
            raise ValueError("axis challenge scale factors must be positive and non-unit")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version: str = "0.1.0"
    global_seed: int = 42
    num_latent_series: int = Field(100, ge=1)
    renderings_per_series: int = Field(2, ge=1)
    curve_grid_size: int = Field(128, ge=2)
    num_workers: int = Field(1, ge=1)
    save_components: bool = True
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    axis_challenges: AxisChallengeConfig = Field(default_factory=AxisChallengeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.model_validate(raw)

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json")
        rendering = payload.get("rendering")
        # Keep config hashes stable when the default backend is chosen implicitly,
        # matching pre-Plotly configs byte for byte.
        if rendering and rendering.get("backend") == "matplotlib":
            rendering.pop("backend", None)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def with_overrides(self, **values: Any) -> DatasetConfig:
        payload = self.model_dump()
        payload.update({key: value for key, value in values.items() if value is not None})
        return DatasetConfig.model_validate(payload)
