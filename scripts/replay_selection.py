"""Replay two-submission policies using solved-task public/private scores."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path


def rank_scores(items: list[dict], key: str) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[key], reverse=True)
    scale = max(1, len(ordered) - 1)
    return {item["file"]: 1.0 - index / scale for index, item in enumerate(ordered)}


def family(filename: str) -> str:
    for name in ("catboost", "lightgbm", "extra_trees", "logistic", "random_forest"):
        if name in filename:
            return name
    return "blend"


def choose_hedged_pair(record: dict, params: tuple[float, float, float]) -> list[dict]:
    alpha, disagreement, family_bonus = params
    items = record["scores"]
    public_rank = rank_scores(items, "test_public")
    cv_rank = rank_scores(items, "cv_auc")
    primary = max(items, key=lambda item: item["test_public"])

    def hedge_score(item: dict) -> float:
        pub = public_rank[item["file"]]
        cv = cv_rank[item["file"]]
        different = family(item["file"]) != family(primary["file"])
        return (
            alpha * pub
            + (1.0 - alpha) * cv
            + disagreement * abs(pub - cv)
            + family_bonus * float(different)
        )

    hedge = max((item for item in items if item is not primary), key=hedge_score)
    return [primary, hedge]


def choose_v3_pair(record: dict) -> list[dict]:
    items = record["scores"]
    primary = max(items, key=lambda item: item["test_public"])
    recommended = items[1] if len(items) > 1 else items[0]
    if recommended is primary or primary["test_public"] - recommended["test_public"] > 0.01:
        recommended = max((item for item in items if item is not primary), key=lambda x: x["test_public"])
    return [primary, recommended]


def pair_score(pair: list[dict]) -> float:
    return max(item["test_private"] for item in pair)


def mean_score(records: list[dict], params: tuple[float, float, float]) -> float:
    return statistics.mean(pair_score(choose_hedged_pair(record, params)) for record in records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    records = json.loads(args.results.read_text(encoding="utf-8"))
    grid = list(itertools.product(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (-0.25, 0.0, 0.25),
        (0.0, 0.1, 0.25),
    ))

    global_best = max(grid, key=lambda params: (mean_score(records, params), -abs(params[0] - 0.5)))
    loo_scores = []
    print("dataset v3_pair loo_pair alpha disagreement family_bonus selections")
    for held_out in records:
        training = [record for record in records if record is not held_out]
        params = max(grid, key=lambda value: (mean_score(training, value), -abs(value[0] - 0.5)))
        v3_score = pair_score(choose_v3_pair(held_out))
        pair = choose_hedged_pair(held_out, params)
        score = pair_score(pair)
        loo_scores.append(score)
        print(
            held_out["dataset"], f"{v3_score:.6f}", f"{score:.6f}",
            *params, ",".join(item["file"] for item in pair),
        )

    public_one = statistics.mean(
        max(record["scores"], key=lambda item: item["test_public"])["test_private"]
        for record in records
    )
    public_two = statistics.mean(
        max(
            item["test_private"]
            for item in sorted(record["scores"], key=lambda x: x["test_public"], reverse=True)[:2]
        )
        for record in records
    )
    v3_pair = statistics.mean(pair_score(choose_v3_pair(record)) for record in records)
    oracle = statistics.mean(max(item["test_private"] for item in record["scores"]) for record in records)
    print("mean_public_one", f"{public_one:.6f}")
    print("mean_public_two", f"{public_two:.6f}")
    print("mean_v3_pair", f"{v3_pair:.6f}")
    print("global_best_params", global_best, f"score={mean_score(records, global_best):.6f}")
    print("mean_loo_pair", f"{statistics.mean(loo_scores):.6f}")
    print("mean_oracle_private", f"{oracle:.6f}")


if __name__ == "__main__":
    main()
