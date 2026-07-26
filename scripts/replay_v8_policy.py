"""Replay v8's public-frontier plus train-CV hedge policy."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def v8_score(record: dict) -> tuple[dict, dict, float]:
    public = sorted(
        record["scores"], key=lambda item: item["test_public"], reverse=True
    )
    cv = sorted(record["scores"], key=lambda item: item["cv_auc"], reverse=True)
    primary = public[0]
    hedge = cv[0] if cv[0]["file"] != primary["file"] else public[1]
    return primary, hedge, max(primary["test_private"], hedge["test_private"])


def v7_score(record: dict, hedges: dict[str, str]) -> float:
    public = sorted(
        record["scores"], key=lambda item: item["test_public"], reverse=True
    )
    primary = public[0]
    hedge_file = hedges.get(record["dataset"])
    if hedge_file:
        hedge = next(item for item in record["scores"] if item["file"] == hedge_file)
        if hedge["file"] == primary["file"]:
            hedge = public[1]
    else:
        hedge = public[1]
    return max(primary["test_private"], hedge["test_private"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--baseline-hedges", type=Path)
    args = parser.parse_args()
    records = json.loads(args.results.read_text(encoding="utf-8"))
    baseline = {}
    hedges = {}
    if args.baseline_results:
        baseline = {
            record["dataset"]: record
            for record in json.loads(args.baseline_results.read_text(encoding="utf-8"))
        }
    if args.baseline_hedges:
        hedges = json.loads(args.baseline_hedges.read_text(encoding="utf-8"))
    scores = []
    deltas = []
    print("dataset primary cv_hedge selected_private baseline_delta")
    for record in records:
        primary, hedge, selected = v8_score(record)
        scores.append(selected)
        delta = None
        if record["dataset"] in baseline:
            delta = selected - v7_score(baseline[record["dataset"]], hedges)
            deltas.append(delta)
        print(
            record["dataset"],
            primary["file"],
            hedge["file"],
            f"{selected:.6f}",
            f"{delta:+.6f}" if delta is not None else "n/a",
        )
    print("mean_selected_private", f"{statistics.mean(scores):.6f}")
    if deltas:
        print("mean_baseline_delta", f"{statistics.mean(deltas):+.6f}")
        print("regressions", sum(delta < -1e-12 for delta in deltas))


if __name__ == "__main__":
    main()
