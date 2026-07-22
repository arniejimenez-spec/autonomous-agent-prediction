"""Compose a full meta-evaluation from a baseline and deterministic overrides."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {record["dataset"]: record for record in load(args.baseline)}
    for path in args.override:
        for record in load(path):
            records[record["dataset"]] = record
    combined = [records[name] for name in sorted(records)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Wrote {len(combined)} datasets to {args.output}")


if __name__ == "__main__":
    main()
