"""Build train-only dataset fingerprints and join them to solved-task outcomes."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
MODEL_RE = re.compile(r"^MODEL (\S+) cv_auc=([0-9.]+)", re.MULTILINE)


def normalized_target(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, str]:
    target = next(col for col in train if col not in test)
    values = sorted(train[target].dropna().unique())
    mapping = {values[0]: 0, values[1]: 1}
    return train[target].map(mapping).astype(int), target


def numeric_signal(frame: pd.DataFrame, y: pd.Series) -> float:
    scores = []
    for col in frame:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.nunique(dropna=True) < 2:
            continue
        values = values.fillna(values.median())
        auc = roc_auc_score(y, values)
        scores.append(max(auc, 1.0 - auc))
    return max(scores, default=0.5)


def categorical_signal(frame: pd.DataFrame, y: pd.Series) -> float:
    if frame.shape[1] == 0:
        return 0.5
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260717)
    best = 0.5
    for col in frame:
        values = frame[col].astype("string").fillna("__MISSING__")
        if values.nunique() < 2:
            continue
        oof = np.zeros(len(frame), dtype=float)
        for fit, valid in folds.split(frame, y):
            global_mean = float(y.iloc[fit].mean())
            stats = pd.DataFrame({"value": values.iloc[fit], "target": y.iloc[fit]}).groupby(
                "value"
            )["target"].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + global_mean * 10.0) / (
                stats["count"] + 10.0
            )
            oof[valid] = values.iloc[valid].map(smoothed).fillna(global_mean)
        auc = roc_auc_score(y, oof)
        best = max(best, auc, 1.0 - auc)
    return best


def fingerprint(dataset: str) -> dict:
    source = ROOT / "data" / dataset
    train = pd.read_csv(source / "train.csv")
    test = pd.read_csv(source / "test.csv")
    y, target = normalized_target(train, test)
    id_cols = [col for col in test if col.lower() in {"id", "row_id"}]
    features = [col for col in test if col not in id_cols]
    x = train[features]
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(x[col])]
    categorical = [col for col in features if col not in numeric]
    low_card_numeric = [
        col for col in numeric
        if x[col].nunique(dropna=True) <= 20
        and np.allclose(
            pd.to_numeric(x[col], errors="coerce").dropna(),
            np.round(pd.to_numeric(x[col], errors="coerce").dropna()),
        )
    ]
    cardinalities = [x[col].nunique(dropna=True) for col in categorical]
    return {
        "dataset": dataset,
        "rows": len(train),
        "features": len(features),
        "numeric": len(numeric),
        "categorical": len(categorical),
        "low_card_numeric": len(low_card_numeric),
        "numeric_fraction": len(numeric) / max(1, len(features)),
        "categorical_view_fraction": (
            len(categorical) + len(low_card_numeric)
        ) / max(1, len(features) + len(low_card_numeric)),
        "missing_fraction": float(x.isna().mean().mean()),
        "positive_rate": float(y.mean()),
        "imbalance": abs(float(y.mean()) - 0.5),
        "median_cat_cardinality": float(np.median(cardinalities)) if cardinalities else 0.0,
        "max_cat_cardinality": float(max(cardinalities, default=0)),
        "numeric_univariate_auc": numeric_signal(x[numeric], y),
        "categorical_univariate_auc": categorical_signal(x[categorical], y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "submissions/07_tree_diversity_automl_v6/meta_results_all.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "submissions/08_fingerprint_routed_automl_v7/fingerprints.csv",
    )
    args = parser.parse_args()
    outcomes = {
        record["dataset"]: record
        for record in json.loads(args.results.read_text(encoding="utf-8"))
    }
    rows = []
    for index in range(1, 17):
        dataset = f"train_{index:02d}"
        item = fingerprint(dataset)
        outcome = outcomes[dataset]
        public_pair = sorted(
            outcome["scores"], key=lambda candidate: candidate["test_public"], reverse=True
        )[:2]
        item.update({
            "selected_private": max(candidate["test_private"] for candidate in public_pair),
            "oracle_private": max(candidate["test_private"] for candidate in outcome["scores"]),
            "public_regret": (
                max(candidate["test_private"] for candidate in outcome["scores"])
                - max(candidate["test_private"] for candidate in public_pair)
            ),
            "best_public_candidate": public_pair[0]["file"],
            "best_private_candidate": max(
                outcome["scores"], key=lambda candidate: candidate["test_private"]
            )["file"],
        })
        item.update({
            f"cv_{name}": float(score)
            for name, score in MODEL_RE.findall(outcome.get("stdout", ""))
        })
        rows.append(item)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
