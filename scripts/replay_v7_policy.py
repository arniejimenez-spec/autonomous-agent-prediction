"""Replay v7's frontier-plus-fingerprint-hedge selection on solved tasks."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--hedges", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.results.read_text(encoding="utf-8"))
    hedges = json.loads(args.hedges.read_text(encoding="utf-8"))
    scores = []
    print("dataset primary hedge selected_private")
    for record in records:
        ordered = sorted(
            record["scores"], key=lambda item: item["test_public"], reverse=True
        )
        primary = ordered[0]
        hedge_name = hedges.get(record["dataset"])
        if hedge_name:
            hedge = next(
                item for item in record["scores"] if item["file"] == hedge_name
            )
            if hedge is primary:
                hedge = ordered[1]
        else:
            hedge = ordered[1]
        selected = max(primary["test_private"], hedge["test_private"])
        scores.append(selected)
        print(
            record["dataset"], primary["file"], hedge["file"], f"{selected:.6f}"
        )
    print("mean_frontier_hedge_private", f"{statistics.mean(scores):.6f}")


if __name__ == "__main__":
    main()
