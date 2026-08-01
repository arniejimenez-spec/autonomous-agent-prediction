"""Audit whether the nominal row identifier carries synthetic target signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def auc_with_direction(y, values) -> tuple[float, float]:
    auc = roc_auc_score(y, values)
    return max(auc, 1.0 - auc), 1.0 if auc >= 0.5 else -1.0


def identifier_views(series: pd.Series) -> dict[str, np.ndarray]:
    text = series.astype(str).str.lower()
    views = {}
    numeric = pd.to_numeric(text, errors="coerce")
    if numeric.notna().all():
        views["numeric"] = numeric.to_numpy(dtype=float)
    if text.str.fullmatch(r"[0-9a-f]+", na=False).all():
        width = int(text.str.len().min())
        for start in range(min(width, 12)):
            views[f"hex_{start}"] = text.str[start].map(
                lambda value: int(value, 16)
            ).to_numpy(dtype=float)
        for label, fragment in (
            ("prefix", text.str[: min(12, width)]),
            ("suffix", text.str[-min(12, width):]),
        ):
            views[label] = fragment.map(lambda value: int(value, 16)).to_numpy(
                dtype=float
            )
    return views


def main() -> None:
    print("dataset train_id_auc test_id_auc train_periodic_auc test_periodic_auc")
    for number in range(1, 17):
        name = f"train_{number:02d}"
        source = ROOT / "data" / name
        train = pd.read_csv(source / "train.csv")
        test = pd.read_csv(source / "test.csv")
        solution = pd.read_csv(source / "solution.csv")
        target = next(column for column in train if column not in test)
        values = sorted(train[target].dropna().unique())
        y_train = train[target].map({values[0]: 0, values[1]: 1}).to_numpy()
        y_test = solution.set_index("row_id").loc[test["row_id"], "target"].to_numpy()
        train_views = identifier_views(train["row_id"])
        test_views = identifier_views(test["row_id"])
        train_scored = []
        for view_name in train_views:
            train_auc, direction = auc_with_direction(y_train, train_views[view_name])
            train_scored.append((train_auc, direction, view_name))
        train_auc, direction, best_view = max(train_scored)
        test_auc = roc_auc_score(y_test, direction * test_views[best_view])
        train_base = train_views[best_view]
        test_base = test_views[best_view]
        periodic = []
        for period in (3.0, 7.0, 17.0, 53.0, 211.0, 997.0):
            for function_name, function in (("sin", np.sin), ("cos", np.cos)):
                train_values = function(train_base / period)
                score, periodic_direction = auc_with_direction(y_train, train_values)
                periodic.append((score, periodic_direction, period, function_name))
        train_periodic, periodic_direction, period, function_name = max(periodic)
        function = np.sin if function_name == "sin" else np.cos
        test_periodic = roc_auc_score(
            y_test, periodic_direction * function(test_base / period)
        )
        print(
            name,
            f"{train_auc:.6f}",
            f"{test_auc:.6f}",
            f"{train_periodic:.6f}",
            f"{test_periodic:.6f}",
            best_view,
        )


if __name__ == "__main__":
    main()
