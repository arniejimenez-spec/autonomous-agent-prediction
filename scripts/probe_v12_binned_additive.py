"""Probe quantile-binned additive classifiers on solved meta-datasets.

This is a repository-level diagnostic.  It may read ``solution.csv`` to assess
whether a representation is worth admitting to the offline competition agent;
the solution files are never bundled with the agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717


def load_frames(dataset: str):
    source = ROOT / "data" / dataset
    train = pd.read_csv(source / "train.csv")
    test = pd.read_csv(source / "test.csv")
    solution = pd.read_csv(source / "solution.csv")
    target = next(column for column in train if column not in test)
    labels = sorted(train[target].dropna().unique())
    y = train[target].map({labels[0]: 0, labels[1]: 1}).astype(int).to_numpy()
    id_col = next(
        (column for column in test if column.lower() in {"id", "row_id"}), None
    )
    features = [column for column in test if column != id_col]
    xtr, xte = train[features].copy(), test[features].copy()
    categorical, numeric = [], []
    for column in features:
        combined = pd.concat([xtr[column], xte[column]], ignore_index=True)
        if pd.api.types.is_numeric_dtype(combined):
            numeric.append(column)
            xtr[column] = pd.to_numeric(xtr[column], errors="coerce")
            xte[column] = pd.to_numeric(xte[column], errors="coerce")
        else:
            categorical.append(column)
            xtr[column] = xtr[column].astype("string").fillna("__MISSING__")
            xte[column] = xte[column].astype("string").fillna("__MISSING__")
    solution_target = next(column for column in solution if column != id_col)
    usage = solution.get("Usage", pd.Series("Private", index=solution.index))
    private = usage.astype(str).str.lower().eq("private").to_numpy()
    truth = solution[solution_target].to_numpy()
    return xtr, xte, y, truth, private, categorical, numeric


def make_model(categorical: list[str], numeric: list[str], bins: int, c: float):
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        (
                            "bins",
                            KBinsDiscretizer(
                                n_bins=bins,
                                encode="onehot",
                                strategy="quantile",
                                subsample=None,
                            ),
                        ),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", min_frequency=2
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("prep", ColumnTransformer(transformers, remainder="drop")),
            (
                "model",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=1200,
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluate(dataset: str):
    xtr, xte, y, truth, private, categorical, numeric = load_frames(dataset)
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(xtr, y)
    )
    records = []
    for bins, c in ((8, 0.10), (16, 0.10), (24, 0.05), (32, 0.03)):
        oof = np.zeros(len(xtr), dtype=float)
        pred = np.zeros(len(xte), dtype=float)
        for fit, valid in folds:
            model = make_model(categorical, numeric, bins, c)
            model.fit(xtr.iloc[fit], y[fit])
            oof[valid] = model.predict_proba(xtr.iloc[valid])[:, 1]
            pred += model.predict_proba(xte)[:, 1] / len(folds)
        records.append(
            {
                "name": f"binned_logistic_{bins}",
                "bins": bins,
                "c": c,
                "cv_auc": float(roc_auc_score(y, oof)),
                "private_auc": float(roc_auc_score(truth[private], pred[private])),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    datasets = args.datasets or [f"train_{number:02d}" for number in range(1, 17)]
    output = []
    print("dataset model cv_auc private_auc")
    for dataset in datasets:
        records = evaluate(dataset)
        output.append({"dataset": dataset, "models": records})
        for record in records:
            print(
                dataset,
                record["name"],
                f'{record["cv_auc"]:.6f}',
                f'{record["private_auc"]:.6f}',
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
