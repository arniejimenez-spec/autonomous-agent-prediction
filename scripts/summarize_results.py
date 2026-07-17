"""Consolidate meta-evaluation shards and print a compact score table."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "submissions/01_robust_automl"


def main() -> None:
    records = []
    for name in ("meta_results_smoke.json", "meta_results_remaining.json"):
        records.extend(json.loads((EXPERIMENT / name).read_text(encoding="utf-8")))
    records.sort(key=lambda item: item["dataset"])
    (EXPERIMENT / "meta_results_all.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    best_test_values, selected_private_values = [], []
    print("dataset best_test best_public private_of_public candidate")
    for record in records:
        best_test = max(item["test_auc"] for item in record["scores"])
        public_pick = max(record["scores"], key=lambda item: item["test_public"])
        best_test_values.append(best_test)
        selected_private_values.append(public_pick["test_private"])
        print(record["dataset"], f"{best_test:.6f}", f"{public_pick['test_public']:.6f}",
              f"{public_pick['test_private']:.6f}", public_pick["file"])
    print("mean_best_test", f"{statistics.mean(best_test_values):.6f}")
    print("mean_public_selected_private", f"{statistics.mean(selected_private_values):.6f}")
    print("range_best_test", f"{min(best_test_values):.6f}", f"{max(best_test_values):.6f}")


if __name__ == "__main__":
    main()
