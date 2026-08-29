"""Command-line entry points for generation, inspection, and validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from chart2data.config import DatasetConfig
from chart2data.dataset.builder import build_dataset
from chart2data.metadata.debug import save_debug_overlay
from chart2data.metadata.serialization import read_jsonl
from chart2data.schema import CanonicalSample
from chart2data.validation.checks import validate_sample, validate_split_leakage


def _load_sample(dataset: Path, sample_id: str) -> CanonicalSample:
    normalized = sample_id.zfill(8)
    for split in ("train", "validation", "test"):
        path = dataset / "jsonl" / f"{split}.jsonl"
        if path.exists():
            for sample in read_jsonl(path):
                if sample.sample_id == normalized or sample.sample_id == sample_id:
                    return sample
    raise FileNotFoundError(f"sample {sample_id!r} was not found in {dataset}")


def _generate(args: argparse.Namespace) -> int:
    config = DatasetConfig.from_yaml(args.config)
    updates = {
        "num_latent_series": args.num_series,
        "renderings_per_series": args.renderings_per_series,
        "global_seed": args.seed,
        "num_workers": args.num_workers,
        "curve_grid_size": args.curve_grid_size,
        "save_components": args.save_components,
    }
    config = config.with_overrides(**updates)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {output}; pass --overwrite")
        shutil.rmtree(output)
    manifest = build_dataset(config, output)
    print(json.dumps(manifest, indent=2))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).resolve()
    sample = _load_sample(dataset, args.sample_id)
    output = (
        Path(args.save_overlay).resolve()
        if args.save_overlay
        else dataset / f"inspect_{sample.sample_id}.png"
    )
    save_debug_overlay(sample, dataset, output)
    summary = {
        "sample_id": sample.sample_id,
        "image": str(dataset / sample.image_path),
        "overlay": str(output),
        "raw_x": sample.data.x,
        "raw_y": sample.data.y,
        "axes": sample.axes.model_dump(mode="json"),
        "plot_area": sample.plot_area.model_dump(mode="json"),
        "rendered_points": [point.model_dump(mode="json") for point in sample.rendered_points],
        "normalized_curve": sample.normalized_curve.model_dump(mode="json"),
        "signal_structure": sample.signal_structure,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).resolve()
    samples = [
        sample
        for split in ("train", "validation", "test")
        for sample in read_jsonl(dataset / "jsonl" / f"{split}.jsonl")
    ]
    failures = {
        sample.sample_id: errors for sample in samples if (errors := validate_sample(sample))
    }
    leakage = validate_split_leakage(samples)
    if failures or leakage:
        print(json.dumps({"sample_failures": failures, "leaked_series": leakage}, indent=2))
        return 1
    print(f"Validated {len(samples)} samples with no consistency or split-leakage errors.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chart2data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate a synthetic dataset")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--num-series", type=int)
    generate.add_argument("--renderings-per-series", type=int)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--num-workers", type=int)
    generate.add_argument("--curve-grid-size", type=int)
    generate.add_argument("--save-components", action=argparse.BooleanOptionalAction, default=None)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(handler=_generate)

    inspect = subparsers.add_parser("inspect", help="inspect canonical truth and create an overlay")
    inspect.add_argument("--dataset", required=True)
    inspect.add_argument("--sample-id", required=True)
    inspect.add_argument("--save-overlay")
    inspect.set_defaults(handler=_inspect)

    validate = subparsers.add_parser("validate", help="validate a generated dataset")
    validate.add_argument("--dataset", required=True)
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
