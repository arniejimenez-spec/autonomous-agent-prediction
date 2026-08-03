"""Replay V14's p01-plus-direct-family selection on saved V13 results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DIRECT_FAMILIES = {
    "catboost", "catboost_shallow", "catboost_ordered", "lightgbm",
    "extra_trees", "random_forest", "histogram", "linear", "spline",
    "target_encoding", "quadratic", "xgboost", "rbf_kernel",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    by_dataset = {}
    for path in args.results:
        for record in json.loads(path.read_text(encoding="utf-8")):
            by_dataset[record["dataset"]] = record

    selected_scores = []
    v13_scores = []
    baseline_deltas = []
    print("dataset direct family p01_private direct_private selected delta_vs_v13")
    for dataset in sorted(by_dataset):
        record = by_dataset[dataset]
        stage1 = record["stage1"]
        hedge = next(item for item in stage1 if item["file"] == "p01.csv")
        direct = [item for item in stage1 if item.get("family") in DIRECT_FAMILIES]
        if not direct:
            direct = [item for item in stage1 if item["file"] != "p01.csv"]
        finalist = max(
            enumerate(direct), key=lambda pair: (pair[1]["public"], -pair[0]),
        )[1]
        selected = max(hedge["private"], finalist["private"])
        selected_scores.append(selected)
        v13_scores.append(record["selected_private"])
        if record.get("baseline_selected_private") is not None:
            baseline_deltas.append(selected - record["baseline_selected_private"])
        print(
            dataset, finalist["file"], finalist.get("family"),
            f"{hedge['private']:.6f}", f"{finalist['private']:.6f}",
            f"{selected:.6f}", f"{selected - record['selected_private']:+.6f}",
        )

    print("mean_selected_private", f"{statistics.mean(selected_scores):.6f}")
    print("mean_delta_vs_v13", f"{statistics.mean(a - b for a, b in zip(selected_scores, v13_scores)):+.6f}")
    if baseline_deltas:
        print("mean_delta_vs_old_baseline", f"{statistics.mean(baseline_deltas):+.6f}")
        print("regressions_vs_old_baseline", sum(delta < -1e-12 for delta in baseline_deltas))


if __name__ == "__main__":
    main()
