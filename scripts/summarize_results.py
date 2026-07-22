"""Consolidate meta-evaluation shards and print a compact score table."""
from __future__ import annotations

import json
import statistics
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="01_robust_automl")
    args = parser.parse_args()
    experiment = ROOT / "submissions" / args.experiment
    records = []
    shards = sorted(experiment.glob("result_*.json"))
    if not shards:
        legacy_shards = [
            experiment / "meta_results_smoke.json",
            experiment / "meta_results_remaining.json",
        ]
        shards = [path for path in legacy_shards if path.is_file()]
    if not shards and (experiment / "meta_results_all.json").is_file():
        shards = [experiment / "meta_results_all.json"]
    if not shards:
        raise FileNotFoundError(f"No result shards found in {experiment}")
    by_dataset = {}
    for shard in shards:
        for record in json.loads(shard.read_text(encoding="utf-8")):
            by_dataset[record["dataset"]] = record
    records = list(by_dataset.values())
    records.sort(key=lambda item: item["dataset"])
    (experiment / "meta_results_all.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    best_test_values, selected_private_values, selected_pair_private_values = [], [], []
    print("dataset best_test best_public private_of_public candidate")
    for record in records:
        best_test = max(item["test_auc"] for item in record["scores"])
        public_pick = max(record["scores"], key=lambda item: item["test_public"])
        public_pair = sorted(record["scores"], key=lambda item: item["test_public"], reverse=True)[:2]
        best_test_values.append(best_test)
        selected_private_values.append(public_pick["test_private"])
        selected_pair_private_values.append(max(item["test_private"] for item in public_pair))
        print(record["dataset"], f"{best_test:.6f}", f"{public_pick['test_public']:.6f}",
              f"{public_pick['test_private']:.6f}", public_pick["file"])
    print("mean_best_test", f"{statistics.mean(best_test_values):.6f}")
    print("mean_public_selected_private", f"{statistics.mean(selected_private_values):.6f}")
    print("mean_top2_public_best_private", f"{statistics.mean(selected_pair_private_values):.6f}")
    print("range_best_test", f"{min(best_test_values):.6f}", f"{max(best_test_values):.6f}")


if __name__ == "__main__":
    main()
