"""Evaluate missing representation model families on solved meta-datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717


def frames(dataset: str):
    source = ROOT / "data" / dataset
    train = pd.read_csv(source / "train.csv")
    test = pd.read_csv(source / "test.csv")
    solution = pd.read_csv(source / "solution.csv")
    target = next(col for col in train if col not in test)
    values = sorted(train[target].dropna().unique())
    y = train[target].map({values[0]: 0, values[1]: 1}).astype(int).to_numpy()
    id_col = next((col for col in test if col.lower() in {"id", "row_id"}), None)
    features = [col for col in test if col != id_col]
    xtr, xte = train[features].copy(), test[features].copy()
    categorical, numeric = [], []
    for col in features:
        combined = pd.concat([xtr[col], xte[col]], ignore_index=True)
        if pd.api.types.is_numeric_dtype(combined):
            numeric.append(col)
            xtr[col] = pd.to_numeric(xtr[col], errors="coerce")
            xte[col] = pd.to_numeric(xte[col], errors="coerce")
        else:
            categorical.append(col)
            xtr[col] = xtr[col].astype("string").fillna("__MISSING__")
            xte[col] = xte[col].astype("string").fillna("__MISSING__")
    return xtr, xte, y, solution, id_col, categorical, numeric


def model_suite(categorical, numeric, rows, fast):
    ordinal = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "enc",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    onehot = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=2, sparse_output=False
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    dimensions = max(1, len(numeric) + len(categorical))
    projected = min(64, max(16, 3 * dimensions))
    result = {
        "lda_shrinkage": Pipeline(
            [
                ("prep", ordinal),
                ("scale", StandardScaler()),
                (
                    "model",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        ),
        "qda_reg_010": Pipeline(
            [
                ("prep", ordinal),
                ("scale", StandardScaler()),
                ("model", QuadraticDiscriminantAnalysis(reg_param=0.10)),
            ]
        ),
        "rotated_hgb": Pipeline(
            [
                ("prep", ordinal),
                ("scale", StandardScaler()),
                (
                    "rotate",
                    GaussianRandomProjection(
                        n_components=projected, random_state=SEED + 301
                    ),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180 if fast else 320,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=max(12, int(np.sqrt(rows) / 2)),
                        l2_regularization=5.0,
                        random_state=SEED + 302,
                    ),
                ),
            ]
        ),
        "rotated_extra_trees": Pipeline(
            [
                ("prep", ordinal),
                ("scale", StandardScaler()),
                (
                    "rotate",
                    GaussianRandomProjection(
                        n_components=projected, random_state=SEED + 311
                    ),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=350 if fast else 600,
                        min_samples_leaf=max(2, int(np.sqrt(rows) / 30)),
                        max_features=0.7,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=SEED + 312,
                    ),
                ),
            ]
        ),
        "rff_logistic": Pipeline(
            [
                ("prep", onehot),
                (
                    "rff",
                    RBFSampler(
                        gamma=1.0 / dimensions,
                        n_components=384 if fast else 768,
                        random_state=SEED + 321,
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.35,
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if rows >= 1000:
        result["mlp_onehot"] = Pipeline(
            [
                ("prep", onehot),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1.0,
                        batch_size=min(256, max(32, rows // 20)),
                        learning_rate_init=0.001,
                        max_iter=180 if fast else 300,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        random_state=SEED + 331,
                    ),
                ),
            ]
        )
    return result


def evaluate(dataset: str, fast: bool, only: set[str] | None = None):
    xtr, xte, y, solution, id_col, categorical, numeric = frames(dataset)
    folds = list(
        StratifiedKFold(
            n_splits=3 if fast else 4, shuffle=True, random_state=SEED
        ).split(xtr, y)
    )
    records = []
    suite = model_suite(categorical, numeric, len(xtr), fast)
    if only:
        suite = {name: model for name, model in suite.items() if name in only}
    for name, model in suite.items():
        oof = np.zeros(len(xtr))
        pred = np.zeros(len(xte))
        for fit, valid in folds:
            fitted = model.fit(xtr.iloc[fit], y[fit])
            oof[valid] = fitted.predict_proba(xtr.iloc[valid])[:, 1]
            pred += fitted.predict_proba(xte)[:, 1] / len(folds)
        prediction = pd.DataFrame({id_col: xte.index if id_col is None else pd.read_csv(
            ROOT / "data" / dataset / "test.csv"
        )[id_col], "target": pred})
        if id_col is None:
            raise ValueError("Expected an identifier column")
        merged = solution.merge(prediction, on=id_col, suffixes=("_true", "_pred"))
        split = {
            usage: roc_auc_score(group["target_true"], group["target_pred"])
            for usage, group in merged.groupby("Usage")
        }
        records.append(
            {
                "name": name,
                "cv_auc": roc_auc_score(y, oof),
                "test_auc": roc_auc_score(
                    merged["target_true"], merged["target_pred"]
                ),
                **{f"test_{key.lower()}": value for key, value in split.items()},
            }
        )
    return {
        "dataset": dataset,
        "rows": len(xtr),
        "features": xtr.shape[1],
        "categorical": len(categorical),
        "numeric": len(numeric),
        "models": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        record = evaluate(dataset, args.fast, set(args.only) if args.only else None)
        results.append(record)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if record["models"]:
            best = max(record["models"], key=lambda item: item["test_auc"])
            print(dataset, best["name"], f"{best['test_auc']:.6f}", flush=True)


if __name__ == "__main__":
    main()
