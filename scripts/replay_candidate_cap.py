"""Replay a compact prefix of an existing candidate frontier."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def candidate_number(filename: str) -> int:
    match = re.match(r"candidate_(\d+)_", filename)
    if not match:
        raise ValueError(f"Cannot infer candidate order from {filename}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--cap", type=int, default=10)
    args = parser.parse_args()
    records = json.loads(args.results.read_text(encoding="utf-8"))
    selected_scores = []
    print("dataset primary hedge selected_private")
    for record in records:
        pool = [
            item
            for item in record["scores"]
            if candidate_number(item["file"]) <= args.cap
        ]
        public = sorted(pool, key=lambda item: item["test_public"], reverse=True)
        cv = sorted(pool, key=lambda item: item["cv_auc"], reverse=True)
        primary = public[0]
        hedge = public[1] if cv[0]["file"] == primary["file"] else cv[0]
        selected = max(primary["test_private"], hedge["test_private"])
        selected_scores.append(selected)
        print(
            record["dataset"],
            primary["file"],
            hedge["file"],
            f"{selected:.6f}",
        )
    print("cap", args.cap)
    print("mean_selected_private", f"{statistics.mean(selected_scores):.6f}")


if __name__ == "__main__":
    main()
