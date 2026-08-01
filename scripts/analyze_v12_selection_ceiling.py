"""Measure the maximum solved-task gain available from selection alone."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def candidate_number(filename: str) -> int:
    match = re.match(r"candidate_(\d+)_", filename)
    if match:
        return int(match.group(1))
    match = re.match(r"p(\d+)\.csv", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer candidate order from {filename}")


def scores_at_cap(record: dict, cap: int) -> list[dict]:
    return [
        item for item in record["scores"]
        if candidate_number(item["file"]) <= cap
    ]


def current_policy(pool: list[dict]) -> float:
    public = sorted(pool, key=lambda item: item["test_public"], reverse=True)
    cv = sorted(pool, key=lambda item: item["cv_auc"], reverse=True)
    hedge = public[1] if public[0]["file"] == cv[0]["file"] else cv[0]
    return max(public[0]["test_private"], hedge["test_private"])


def top_two_public(pool: list[dict]) -> float:
    pair = sorted(pool, key=lambda item: item["test_public"], reverse=True)[:2]
    return max(item["test_private"] for item in pair)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--caps", nargs="+", type=int, default=[10, 12, 14, 20, 28])
    args = parser.parse_args()
    records = json.loads(args.results.read_text(encoding="utf-8"))
    print("cap current top2_public private_oracle oracle_headroom")
    for cap in args.caps:
        pools = [scores_at_cap(record, cap) for record in records]
        current = statistics.mean(current_policy(pool) for pool in pools)
        top2 = statistics.mean(top_two_public(pool) for pool in pools)
        oracle = statistics.mean(
            max(item["test_private"] for item in pool) for pool in pools
        )
        print(
            cap,
            f"{current:.6f}",
            f"{top2:.6f}",
            f"{oracle:.6f}",
            f"{oracle - current:+.6f}",
        )


if __name__ == "__main__":
    main()
