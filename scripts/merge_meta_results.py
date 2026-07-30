"""Merge one-record meta-evaluation JSON files into dataset order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for source in args.inputs:
        payload = json.loads(source.read_text(encoding="utf-8"))
        records.extend(payload if isinstance(payload, list) else [payload])
    datasets = [record["dataset"] for record in records]
    duplicates = sorted(
        dataset for dataset in set(datasets) if datasets.count(dataset) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate datasets: {duplicates}")
    records.sort(key=lambda record: record["dataset"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
