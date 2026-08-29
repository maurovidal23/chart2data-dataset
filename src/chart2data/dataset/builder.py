"""End-to-end deterministic dataset generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from chart2data import __version__
from chart2data.config import DatasetConfig
from chart2data.dataset.splits import assign_splits, split_counts
from chart2data.dataset.writer import write_manifest, write_split_metadata
from chart2data.metadata.canonical import build_canonical_sample
from chart2data.random import make_rng
from chart2data.rendering import get_renderer, resolve_backend
from chart2data.schema import CanonicalSample
from chart2data.signals.generator import generate_signal
from chart2data.validation.checks import validate_sample, validate_split_leakage


def _package_versions(*names: str) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _challenge_details(index: int, config: DatasetConfig) -> tuple[int, float, str | None]:
    challenge = config.axis_challenges
    paired_series = 2 * int((config.num_latent_series * challenge.fraction) // 2)
    if not challenge.enabled or index >= paired_series:
        return index, 1.0, None
    base_index = index - index % 2
    group_id = f"axis_scale_{base_index // 2:08d}"
    if index % 2 == 0:
        return base_index, 1.0, group_id
    factor_rng = make_rng(config.global_seed, group_id, "scale-factor")
    return base_index, float(factor_rng.choice(challenge.scale_factors)), group_id


def _generate_series(
    index: int, split: str, config_payload: dict[str, Any], output_text: str
) -> list[dict[str, Any]]:
    config = DatasetConfig.model_validate(config_payload)
    output = Path(output_text)
    latent_id = f"series_{index:08d}"
    source_index, scale_factor, challenge_group_id = _challenge_details(index, config)
    source_id = f"series_{source_index:08d}"
    signal_rng = make_rng(config.global_seed, source_id, "latent")
    signal = generate_signal(signal_rng, config.signal)
    y = signal.y * scale_factor
    records: list[dict[str, Any]] = []
    for rendering_index in range(config.renderings_per_series):
        rendering_id = f"render_{rendering_index:02d}"
        sample_number = index * config.renderings_per_series + rendering_index
        sample_id = f"{sample_number:08d}"
        relative_image = Path("images") / split / f"{sample_id}.png"
        render_key = challenge_group_id or latent_id
        render_rng = make_rng(config.global_seed, render_key, rendering_id)
        backend_selection_rng = make_rng(config.global_seed, render_key, rendering_id, "backend")
        backend = resolve_backend(config.rendering, backend_selection_rng)
        renderer = get_renderer(config.rendering, backend)
        rendered = renderer.render(signal.x, y, output / relative_image, render_rng)
        sample = build_canonical_sample(
            sample_id=sample_id,
            latent_series_id=latent_id,
            rendering_id=rendering_id,
            split=split,
            image_path=relative_image.as_posix(),
            x=signal.x,
            y=y,
            rendered=rendered,
            curve_grid_size=config.curve_grid_size,
            signal_structure=signal.structure if config.save_components else None,
            challenge_group_id=challenge_group_id,
        )
        errors = validate_sample(sample)
        if errors:
            raise RuntimeError(f"invalid generated sample {sample_id}: {errors}")
        records.append(sample.model_dump(mode="json"))
    return records


def build_dataset(config: DatasetConfig, output: str | Path) -> dict[str, object]:
    """Generate images, canonical metadata, and a reproducibility manifest."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    assignments = assign_splits(config.num_latent_series, config.split_ratios, config.global_seed)
    challenge = config.axis_challenges
    paired_series = 2 * int((config.num_latent_series * challenge.fraction) // 2)
    if challenge.enabled:
        for index in range(0, paired_series, 2):
            assignments[f"series_{index + 1:08d}"] = assignments[f"series_{index:08d}"]
    tasks = [
        (
            index,
            assignments[f"series_{index:08d}"],
            config.model_dump(mode="json"),
            str(output_path),
        )
        for index in range(config.num_latent_series)
    ]
    if config.num_workers == 1:
        nested = [_generate_series(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            nested = list(executor.map(_generate_series_star, tasks))
    samples = [CanonicalSample.model_validate(record) for group in nested for record in group]
    samples.sort(key=lambda sample: sample.sample_id)
    leakage = validate_split_leakage(samples)
    if leakage:
        raise RuntimeError(f"latent series leaked across splits: {leakage[:5]}")
    write_split_metadata(samples, output_path)
    counts = split_counts(assignments)
    manifest: dict[str, object] = {
        "dataset_version": config.dataset_version,
        "global_seed": config.global_seed,
        "num_latent_series": config.num_latent_series,
        "renderings_per_series": config.renderings_per_series,
        "num_images": len(samples),
        "curve_grid_size": config.curve_grid_size,
        "splits": counts,
        "split_images": {
            key: value * config.renderings_per_series for key, value in counts.items()
        },
        "config_hash": config.stable_hash(),
        "generator_version": __version__,
        "renderer_environments": _package_versions("matplotlib", "plotly", "kaleido"),
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
    }
    write_manifest(manifest, output_path)
    return manifest


def _generate_series_star(arguments: tuple[int, str, dict[str, Any], str]) -> list[dict[str, Any]]:
    return _generate_series(*arguments)
