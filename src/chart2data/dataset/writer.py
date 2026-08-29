"""Dataset layout and manifest persistence."""

from __future__ import annotations

import json
from pathlib import Path

from chart2data.metadata.serialization import write_jsonl, write_parquet
from chart2data.schema import CanonicalSample


def write_split_metadata(samples: list[CanonicalSample], output: Path) -> None:
    grouped = {
        split: [sample for sample in samples if sample.split == split]
        for split in ("train", "validation", "test")
    }
    for split, records in grouped.items():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        write_jsonl(records, output / "jsonl" / f"{split}.jsonl")
        if records:
            write_parquet(records, output / "metadata" / f"{split}.parquet")


def write_manifest(manifest: dict[str, object], output: Path) -> None:
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
