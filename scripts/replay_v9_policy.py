"""Compare v9's explicit historical hedge with the v8.1 ten-candidate policy."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def selected_score(record: dict, cap: int | None = None) -> tuple[dict, dict, float]:
    pool = record["scores"][:cap] if cap else record["scores"]
    public = sorted(pool, key=lambda item: item["test_public"], reverse=True)
    hedge_file = record.get("cv_hedge_file")
    if hedge_file:
        hedge = next(item for item in pool if item["file"] == hedge_file)
    else:
        hedge = max(pool, key=lambda item: item["cv_auc"])
    primary = public[0]
    if hedge["file"] == primary["file"]:
        hedge = public[1]
    return primary, hedge, max(primary["test_private"], hedge["test_private"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--baseline-cap", type=int, default=10)
    args = parser.parse_args()

    records = json.loads(args.results.read_text(encoding="utf-8"))
    baseline = {
        record["dataset"]: record
        for record in json.loads(args.baseline_results.read_text(encoding="utf-8"))
    }
    scores, deltas = [], []
    print("dataset primary hedge selected_private baseline delta")
    for record in records:
        primary, hedge, selected = selected_score(record)
        _, _, prior = selected_score(
            baseline[record["dataset"]], cap=args.baseline_cap
        )
        delta = selected - prior
        scores.append(selected)
        deltas.append(delta)
        print(
            record["dataset"],
            primary["file"],
            hedge["file"],
            f"{selected:.6f}",
            f"{prior:.6f}",
            f"{delta:+.6f}",
        )
    print("mean_selected_private", f"{statistics.mean(scores):.6f}")
    print(
        "baseline_mean_selected_private",
        f"{statistics.mean(score - delta for score, delta in zip(scores, deltas)):.6f}",
    )
    print("mean_delta", f"{statistics.mean(deltas):+.6f}")
    print("regressions", sum(delta < -1e-12 for delta in deltas))


if __name__ == "__main__":
    main()
