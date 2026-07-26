"""Evaluate a leakage-safe dataset-level meta-router and export its coefficients.

The outer validation unit is an entire mini-competition. For every held-out
dataset, the ridge ranker is fitted only on candidates from the other datasets.
Public test scores are used solely to identify the first portfolio slot; the
meta-router sees train-only fingerprints and cross-validation diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TOKEN_NAMES = (
    "blend",
    "broad",
    "top2",
    "weighted",
    "individual",
    "extra_trees",
    "random_forest",
    "logistic",
    "quadratic",
    "target_encoded",
    "xgboost",
    "catboost",
    "cat_d4",
    "cat_d6",
    "cat_d8",
    "ordered",
    "lightgbm",
    "v21",
    "v3",
    "v4",
    "v5",
    "v6",
    "safe",
)
DATASET_FEATURES = (
    "log_rows",
    "log_features",
    "numeric_fraction",
    "categorical_view_fraction",
    "missing_fraction",
    "imbalance",
    "numeric_signal",
    "categorical_signal",
    "best_model_cv",
    "model_cv_spread",
    "logistic_gap",
    "catboost_gap",
    "lightgbm_gap",
    "tree_gap",
)
CANDIDATE_FEATURES = (
    "candidate_cv",
    "cv_delta_best",
    "cv_rank_fraction",
)


def clean_name(filename: str) -> str:
    name = re.sub(r"^candidate_\d+_", "", filename)
    return re.sub(r"\.csv$", "", name)


def candidate_tokens(filename: str) -> dict[str, float]:
    name = clean_name(filename).lower()
    individual_names = (
        "extra_trees",
        "random_forest",
        "logistic",
        "quadratic_logistic",
        "target_encoded_logistic",
        "xgboost",
        "catboost_d4_smooth",
        "catboost_d4_smooth_seed_b",
        "catboost_d6",
        "catboost_d8",
        "catboost_ordered_d5",
        "catboost_ordered_d5_seed_b",
        "lightgbm",
    )
    return {
        "blend": float("blend" in name),
        "broad": float("broad" in name),
        "top2": float("top2" in name),
        "weighted": float("weighted" in name),
        "individual": float(name in individual_names),
        "extra_trees": float("extra_trees" in name),
        "random_forest": float("random_forest" in name),
        "logistic": float(
            "logistic" in name
            and "quadratic" not in name
            and "target_encoded" not in name
        ),
        "quadratic": float("quadratic" in name),
        "target_encoded": float("target_" in name),
        "xgboost": float("xgboost" in name),
        "catboost": float("catboost" in name),
        "cat_d4": float("d4" in name),
        "cat_d6": float("d6" in name),
        "cat_d8": float("d8" in name),
        "ordered": float("ordered" in name),
        "lightgbm": float("lightgbm" in name),
        "v21": float("v21" in name),
        "v3": float(re.search(r"(^|_)v3_", name) is not None),
        "v4": float("v4" in name),
        "v5": float("v5" in name),
        "v6": float("v6" in name),
        "safe": float("safe" in name),
    }


def available_cv(row: pd.Series, name: str) -> float | None:
    value = row.get(f"cv_{name}", np.nan)
    if pd.isna(value):
        return None
    return float(value)


def dataset_vector(row: pd.Series) -> dict[str, float]:
    model_names = (
        "extra_trees",
        "random_forest",
        "logistic",
        "quadratic_logistic",
        "target_encoded_logistic",
        "xgboost",
        "catboost_d4_smooth",
        "catboost_d6",
        "catboost_d8",
        "catboost_ordered_d5",
        "lightgbm",
    )
    scores = {
        name: score
        for name in model_names
        if (score := available_cv(row, name)) is not None
    }
    best = max(scores.values(), default=0.5)
    worst = min(scores.values(), default=0.5)

    def gap(*names: str) -> float:
        found = [scores[name] for name in names if name in scores]
        return max(found, default=0.5) - best

    return {
        "log_rows": math.log1p(float(row["rows"])),
        "log_features": math.log1p(float(row["features"])),
        "numeric_fraction": float(row["numeric_fraction"]),
        "categorical_view_fraction": float(row["categorical_view_fraction"]),
        "missing_fraction": float(row["missing_fraction"]),
        "imbalance": float(row["imbalance"]),
        "numeric_signal": float(row["numeric_univariate_auc"]) - 0.5,
        "categorical_signal": float(row["categorical_univariate_auc"]) - 0.5,
        "best_model_cv": best - 0.5,
        "model_cv_spread": best - worst,
        "logistic_gap": gap("logistic", "target_encoded_logistic"),
        "catboost_gap": gap(
            "catboost_d4_smooth",
            "catboost_d6",
            "catboost_d8",
            "catboost_ordered_d5",
        ),
        "lightgbm_gap": gap("lightgbm"),
        "tree_gap": gap("extra_trees", "random_forest", "xgboost"),
    }


def feature_names() -> list[str]:
    names = list(DATASET_FEATURES) + list(CANDIDATE_FEATURES) + [
        f"token_{token}" for token in TOKEN_NAMES
    ]
    interaction_inputs = (
        "log_rows",
        "log_features",
        "numeric_fraction",
        "categorical_view_fraction",
        "missing_fraction",
        "numeric_signal",
        "categorical_signal",
        "model_cv_spread",
        "logistic_gap",
        "catboost_gap",
        "lightgbm_gap",
        "tree_gap",
        "cv_delta_best",
    )
    names.extend(
        f"token_{token}_x_{feature}"
        for token in TOKEN_NAMES
        for feature in interaction_inputs
    )
    return names


def candidate_vector(
    dataset: dict[str, float],
    score: dict,
    best_cv: float,
    cv_rank: int,
    count: int,
) -> np.ndarray:
    tokens = candidate_tokens(score["file"])
    candidate = {
        "candidate_cv": float(score["cv_auc"]) - 0.5,
        "cv_delta_best": float(score["cv_auc"]) - best_cv,
        "cv_rank_fraction": cv_rank / max(1, count - 1),
    }
    values = [dataset[name] for name in DATASET_FEATURES]
    values.extend(candidate[name] for name in CANDIDATE_FEATURES)
    values.extend(tokens[name] for name in TOKEN_NAMES)
    interaction_inputs = (
        "log_rows",
        "log_features",
        "numeric_fraction",
        "categorical_view_fraction",
        "missing_fraction",
        "numeric_signal",
        "categorical_signal",
        "model_cv_spread",
        "logistic_gap",
        "catboost_gap",
        "lightgbm_gap",
        "tree_gap",
        "cv_delta_best",
    )
    combined = dataset | candidate
    values.extend(
        tokens[token] * combined[feature]
        for token in TOKEN_NAMES
        for feature in interaction_inputs
    )
    return np.asarray(values, dtype=float)


def build_rows(results: list[dict], fingerprints: pd.DataFrame) -> list[dict]:
    fingerprints = fingerprints.set_index("dataset")
    rows = []
    for record in results:
        dataset = record["dataset"]
        dataset_features = dataset_vector(fingerprints.loc[dataset])
        scores = record["scores"]
        ordered_cv = sorted(scores, key=lambda item: item["cv_auc"], reverse=True)
        ranks = {item["file"]: rank for rank, item in enumerate(ordered_cv)}
        best_cv = float(ordered_cv[0]["cv_auc"])
        private_mean = float(np.mean([item["test_private"] for item in scores]))
        for score in scores:
            rows.append(
                {
                    "dataset": dataset,
                    "file": score["file"],
                    "x": candidate_vector(
                        dataset_features,
                        score,
                        best_cv,
                        ranks[score["file"]],
                        len(scores),
                    ),
                    "target": float(score["test_private"]) - private_mean,
                    "private": float(score["test_private"]),
                    "public": float(score["test_public"]),
                }
            )
    return rows


def fit_ranker(rows: list[dict], alpha: float) -> tuple[StandardScaler, Ridge]:
    x = np.vstack([row["x"] for row in rows])
    y = np.asarray([row["target"] for row in rows])
    dataset_counts = {}
    for row in rows:
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
    weights = np.asarray([1.0 / dataset_counts[row["dataset"]] for row in rows])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    model = Ridge(alpha=alpha)
    model.fit(scaled, y, sample_weight=weights)
    return scaler, model


def predict_rows(
    rows: list[dict], scaler: StandardScaler, model: Ridge
) -> np.ndarray:
    return model.predict(scaler.transform(np.vstack([row["x"] for row in rows])))


def choose_distinct(rows: list[dict], predictions: np.ndarray, first_file: str) -> dict:
    ordered = sorted(
        zip(rows, predictions, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return next(row for row, _ in ordered if row["file"] != first_file)


def evaluate_alpha(rows: list[dict], alpha: float) -> tuple[float, list[dict]]:
    datasets = sorted({row["dataset"] for row in rows})
    details = []
    for held_out in datasets:
        train_rows = [row for row in rows if row["dataset"] != held_out]
        test_rows = [row for row in rows if row["dataset"] == held_out]
        scaler, model = fit_ranker(train_rows, alpha)
        predictions = predict_rows(test_rows, scaler, model)
        public_order = sorted(test_rows, key=lambda row: row["public"], reverse=True)
        first = public_order[0]
        hedge = choose_distinct(test_rows, predictions, first["file"])
        selected = max(first["private"], hedge["private"])
        details.append(
            {
                "dataset": held_out,
                "primary": first["file"],
                "hedge": hedge["file"],
                "selected_private": selected,
                "public_top2_private": max(
                    public_order[0]["private"], public_order[1]["private"]
                ),
                "cv_top_private": max(
                    first["private"],
                    max(test_rows, key=lambda row: row["x"][len(DATASET_FEATURES)])[
                        "private"
                    ],
                ),
            }
        )
    return float(np.mean([item["selected_private"] for item in details])), details


def evaluate_nested(
    rows: list[dict], alphas: tuple[float, ...]
) -> tuple[float, list[dict]]:
    """Choose ridge regularization inside each outer dataset holdout."""
    datasets = sorted({row["dataset"] for row in rows})
    details = []
    for held_out in datasets:
        train_rows = [row for row in rows if row["dataset"] != held_out]
        test_rows = [row for row in rows if row["dataset"] == held_out]
        inner_scores = [
            (evaluate_alpha(train_rows, alpha)[0], alpha) for alpha in alphas
        ]
        _, alpha = max(inner_scores)
        scaler, model = fit_ranker(train_rows, alpha)
        predictions = predict_rows(test_rows, scaler, model)
        public_order = sorted(test_rows, key=lambda row: row["public"], reverse=True)
        first = public_order[0]
        hedge = choose_distinct(test_rows, predictions, first["file"])
        details.append(
            {
                "dataset": held_out,
                "alpha": alpha,
                "primary": first["file"],
                "hedge": hedge["file"],
                "selected_private": max(first["private"], hedge["private"]),
            }
        )
    return float(np.mean([item["selected_private"] for item in details])), details


def export_ranker(
    rows: list[dict],
    alpha: float,
    destination: Path,
    validation_score: float,
) -> None:
    scaler, model = fit_ranker(rows, alpha)
    payload = {
        "version": 1,
        "method": "group-held-out ridge candidate ranker",
        "alpha": alpha,
        "validation_selected_private": validation_score,
        "feature_names": feature_names(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "token_names": list(TOKEN_NAMES),
        "dataset_features": list(DATASET_FEATURES),
        "candidate_features": list(CANDIDATE_FEATURES),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "submissions/08_fingerprint_routed_automl_v7/meta_results_all.json",
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=ROOT / "submissions/08_fingerprint_routed_automl_v7/fingerprints.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "submissions/09_meta_routed_automl_v8"
            / "meta_router_exploratory.json"
        ),
    )
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    fingerprints = pd.read_csv(args.fingerprints)
    rows = build_rows(results, fingerprints)
    alphas = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    trials = []
    for alpha in alphas:
        score, details = evaluate_alpha(rows, alpha)
        trials.append((score, alpha, details))
        print(f"alpha={alpha:6.1f} loodo_selected_private={score:.6f}")
    score, alpha, details = max(trials)
    print(f"\nSELECTED alpha={alpha} loodo_selected_private={score:.6f}")
    print("dataset primary hedge selected_private public_top2_private cv_top_private")
    for item in details:
        print(
            item["dataset"],
            item["primary"],
            item["hedge"],
            f"{item['selected_private']:.6f}",
            f"{item['public_top2_private']:.6f}",
            f"{item['cv_top_private']:.6f}",
        )
    print(
        "mean_public_top2_private",
        f"{np.mean([item['public_top2_private'] for item in details]):.6f}",
    )
    print(
        "mean_cv_top_private",
        f"{np.mean([item['cv_top_private'] for item in details]):.6f}",
    )
    nested_score, nested_details = evaluate_nested(rows, alphas)
    print(
        "nested_loodo_selected_private",
        f"{nested_score:.6f}",
        "alphas=" + ",".join(
            f"{item['dataset']}:{item['alpha']}" for item in nested_details
        ),
    )
    export_ranker(rows, alpha, args.output, score)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
