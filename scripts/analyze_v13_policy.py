"""Replay V13 replacement thresholds without retraining any models."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def selection(record: dict, threshold: float) -> tuple[str, str, float, float]:
    stage1 = record["stage1"]
    refined = record["refined"]
    stage1_ranked = sorted(
        enumerate(stage1), key=lambda pair: (-pair[1]["public"], pair[0]),
    )
    refined_ranked = sorted(
        enumerate(refined), key=lambda pair: (-pair[1]["public"], pair[0]),
    )
    stage1_leader = stage1_ranked[0][1]
    refined_leader = refined_ranked[0][1]
    gain = refined_leader["public"] - stage1_leader["public"]
    primary = refined_leader if gain >= threshold else stage1_leader
    hedge = next(item for item in stage1 if item["file"] == "p01.csv")
    if primary["file"] == hedge["file"]:
        hedge = next(item for _, item in stage1_ranked if item["file"] != primary["file"])
    return primary["file"], hedge["file"], max(primary["private"], hedge["private"]), gain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001],
    )
    args = parser.parse_args()
    records = []
    for path in args.results:
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    # Later files intentionally replace earlier diagnostics for the same task.
    records = list({record["dataset"]: record for record in records}.values())
    records.sort(key=lambda record: record["dataset"])

    print("threshold mean_private mean_delta regressions refined_count")
    for threshold in args.thresholds:
        selected = [selection(record, threshold) for record in records]
        scores = [item[2] for item in selected]
        deltas = [
            score - record["baseline_selected_private"]
            for score, record in zip(scores, records)
            if record.get("baseline_selected_private") is not None
        ]
        refined_count = sum(primary.startswith("r") for primary, _, _, _ in selected)
        print(
            f"{threshold:.5f}", f"{statistics.mean(scores):.6f}",
            f"{statistics.mean(deltas):+.6f}" if deltas else "n/a",
            sum(delta < -1e-12 for delta in deltas), refined_count,
        )

    best_threshold = max(
        args.thresholds,
        key=lambda value: statistics.mean(selection(record, value)[2] for record in records),
    )
    print("\nbest_threshold", f"{best_threshold:.5f}")
    print("dataset primary hedge selected_private delta public_refine_gain")
    for record in records:
        primary, hedge, score, gain = selection(record, best_threshold)
        delta = score - record["baseline_selected_private"]
        print(
            record["dataset"], primary, hedge, f"{score:.6f}",
            f"{delta:+.6f}", f"{gain:+.6f}",
        )


if __name__ == "__main__":
    main()
