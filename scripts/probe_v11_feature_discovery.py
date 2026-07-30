"""Probe equation-oriented feature families on the sixteen solved meta-datasets.

This is deliberately separate from the submitted agent.  A feature family must
first show train-only routing signal and solved-test transfer before it is
allowed into the runtime portfolio.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717
warnings.filterwarnings("ignore")


class EquationFeatures(BaseEstimator, TransformerMixin):
    """Fold-safe robust scaling plus common synthetic-equation primitives."""

    def __init__(self, relations: bool = False):
        self.relations = relations

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        self.medians_ = np.nanmedian(values, axis=0)
        filled = np.where(np.isnan(values), self.medians_, values)
        self.centers_ = np.nanmedian(filled, axis=0)
        q25 = np.nanpercentile(filled, 25, axis=0)
        q75 = np.nanpercentile(filled, 75, axis=0)
        scale = q75 - q25
        fallback = np.nanstd(filled, axis=0)
        self.scales_ = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float)
        missing = np.isnan(values).astype(float)
        filled = np.where(np.isnan(values), self.medians_, values)
        z = np.clip((filled - self.centers_) / self.scales_, -12.0, 12.0)
        blocks = [
            z,
            np.abs(z),
            np.square(z),
            np.sign(z) * np.sqrt(np.abs(z)),
            np.sign(z) * np.log1p(np.abs(z)),
            np.tanh(z),
            missing,
        ]
        if self.relations and z.shape[1] >= 2:
            relations = []
            for left in range(z.shape[1] - 1):
                a = z[:, left]
                for right in range(left + 1, z.shape[1]):
                    b = z[:, right]
                    product = np.clip(a * b, -30.0, 30.0)
                    relations.extend(
                        [
                            product,
                            np.abs(a - b),
                            a / (1.0 + np.abs(b)),
                            b / (1.0 + np.abs(a)),
                            np.sqrt(np.square(a) + np.square(b)),
                            np.sin(np.clip(product, -np.pi, np.pi)),
                        ]
                    )
            blocks.append(np.column_stack(relations))
        return np.column_stack(blocks)


def load_frames(dataset: str):
    source = ROOT / "data" / dataset
    train = pd.read_csv(source / "train.csv")
    test = pd.read_csv(source / "test.csv")
    solution = pd.read_csv(source / "solution.csv")
    target = next(col for col in train if col not in test)
    values = sorted(train[target].dropna().unique())
    y = train[target].map({values[0]: 0, values[1]: 1}).astype(int).to_numpy()
    id_col = next(col for col in test if col.lower() in {"id", "row_id"})
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
    return source, xtr, xte, y, solution, id_col, categorical, numeric


def model_suite(categorical, numeric, rows):
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=2)

    def equation_model(relations: bool, c_value: float):
        prep = ColumnTransformer(
            [
                ("num", EquationFeatures(relations=relations), numeric),
                ("cat", onehot, categorical),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("prep", prep),
                ("scale", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        max_iter=1600,
                        class_weight="balanced",
                        solver="liblinear" if rows < 2000 else "lbfgs",
                    ),
                ),
            ]
        )

    result = {
        "unary_equations": equation_model(relations=False, c_value=0.06),
    }
    if 2 <= len(numeric) <= 30 and rows <= 30000:
        result["relation_equations"] = equation_model(relations=True, c_value=0.025)
    if numeric and rows <= 30000:
        binned = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                            (
                                "bins",
                                KBinsDiscretizer(
                                    n_bins=10,
                                    encode="onehot",
                                    strategy="quantile",
                                    subsample=None,
                                ),
                            ),
                        ]
                    ),
                    numeric,
                ),
                ("cat", onehot, categorical),
            ],
            remainder="drop",
        )
        result["quantile_bins"] = Pipeline(
            [
                ("prep", binned),
                (
                    "model",
                    LogisticRegression(
                        C=0.15,
                        max_iter=1200,
                        class_weight="balanced",
                        solver="liblinear" if rows < 2000 else "lbfgs",
                    ),
                ),
            ]
        )
    return result


def score_test(source, test, solution, id_col, prediction):
    pred = pd.DataFrame({id_col: test[id_col], "target": prediction})
    merged = solution.merge(pred, on=id_col, suffixes=("_true", "_pred"))
    splits = {
        usage.lower(): roc_auc_score(group["target_true"], group["target_pred"])
        for usage, group in merged.groupby("Usage")
    }
    return roc_auc_score(merged["target_true"], merged["target_pred"]), splits


def evaluate(dataset: str, only: set[str] | None = None):
    source, xtr, xte, y, solution, id_col, categorical, numeric = load_frames(dataset)
    raw_test = pd.read_csv(source / "test.csv")
    seeds = (SEED, SEED + 991)
    models = model_suite(categorical, numeric, len(xtr))
    if only:
        models = {name: model for name, model in models.items() if name in only}
    records = []
    for name, model in models.items():
        seed_oofs, seed_preds, seed_scores = [], [], []
        for seed in seeds:
            oof = np.zeros(len(xtr))
            prediction = np.zeros(len(xte))
            folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            for fit, valid in folds.split(xtr, y):
                fitted = model.fit(xtr.iloc[fit], y[fit])
                oof[valid] = fitted.predict_proba(xtr.iloc[valid])[:, 1]
                prediction += fitted.predict_proba(xte)[:, 1] / folds.n_splits
            seed_oofs.append(oof)
            seed_preds.append(prediction)
            seed_scores.append(roc_auc_score(y, oof))
        mean_prediction = np.mean(seed_preds, axis=0)
        test_auc, split_auc = score_test(
            source, raw_test, solution, id_col, mean_prediction
        )
        records.append(
            {
                "name": name,
                "cv_auc_mean": float(np.mean(seed_scores)),
                "cv_auc_min": float(np.min(seed_scores)),
                "cv_auc_spread": float(np.max(seed_scores) - np.min(seed_scores)),
                "test_auc": test_auc,
                **{f"test_{key}": value for key, value in split_auc.items()},
            }
        )
    return {
        "dataset": dataset,
        "rows": len(xtr),
        "features": xtr.shape[1],
        "numeric": len(numeric),
        "categorical": len(categorical),
        "models": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        record = evaluate(dataset, set(args.only) if args.only else None)
        results.append(record)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if record["models"]:
            best = max(record["models"], key=lambda item: item["test_auc"])
            print(
                dataset,
                best["name"],
                f"cv={best['cv_auc_mean']:.6f}",
                f"test={best['test_auc']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
