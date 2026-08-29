"""Lossless JSONL and Arrow/Parquet canonical serialization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chart2data.schema import CanonicalSample


def write_jsonl(samples: Iterable[CanonicalSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json(exclude_none=False) + "\n")


def read_jsonl(path: Path) -> list[CanonicalSample]:
    with path.open("r", encoding="utf-8") as handle:
        return [CanonicalSample.model_validate_json(line) for line in handle if line.strip()]


def _arrow_record(sample: CanonicalSample) -> dict[str, object]:
    record = sample.model_dump(mode="json")
    structure = record.pop("signal_structure", None)
    record["signal_structure_json"] = (
        json.dumps(structure, separators=(",", ":"), sort_keys=True)
        if structure is not None
        else None
    )
    return record


def write_parquet(samples: Iterable[CanonicalSample], path: Path) -> None:
    records = [_arrow_record(sample) for sample in samples]
    if not records:
        raise ValueError("cannot infer a Parquet schema from an empty split")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


def read_parquet(path: Path) -> list[CanonicalSample]:
    records = pq.read_table(path).to_pylist()
    samples: list[CanonicalSample] = []
    for record in records:
        encoded = record.pop("signal_structure_json")
        record["signal_structure"] = json.loads(encoded) if encoded is not None else None
        samples.append(CanonicalSample.model_validate(record))
    return samples
