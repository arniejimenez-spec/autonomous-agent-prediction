#!/usr/bin/env python3
"""Budget-aware mixed-type AutoML for the Kaggle-in-Kaggle sandbox."""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 20260717


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / (len(values) + 1.0)


def find_columns(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame):
    target_candidates = [c for c in train.columns if c not in test.columns]
    if len(target_candidates) != 1:
        target_candidates = [c for c in sample.columns if c not in test.columns or c in train.columns]
    target = "target" if "target" in target_candidates else target_candidates[-1]
    pred_cols = [c for c in sample.columns if c != target]
    id_col = pred_cols[0] if pred_cols else None
    features = [c for c in test.columns if c != id_col]
    return target, id_col, features


def normalize_target(series: pd.Series):
    vals = list(pd.Series(series.dropna().unique()).sort_values())
    if len(vals) != 2:
        raise ValueError(f"Expected a binary target, found {vals}")
    mapping = {vals[0]: 0, vals[1]: 1}
    return series.map(mapping).astype(int).to_numpy(), mapping


def prepare_frames(train, test, features):
    xtr = train[features].copy()
    xte = test[features].copy()
    cat_cols = []
    num_cols = []
    for col in list(features):
        combined = pd.concat([xtr[col], xte[col]], ignore_index=True)
        if not pd.api.types.is_numeric_dtype(combined) or pd.api.types.is_bool_dtype(combined):
            # Preserve nominal handling, but recover explicit ord_0, ord_1, ... ordering.
            cat_cols.append(col)
            xtr[col] = xtr[col].astype("string").fillna("__MISSING__")
            xte[col] = xte[col].astype("string").fillna("__MISSING__")
            nonmissing = combined.dropna().astype(str)
            extracted = nonmissing.str.extract(r"^ord_(-?\d+(?:\.\d+)?)$", expand=False)
            if len(nonmissing) and extracted.notna().mean() >= 0.8:
                ordered_col = f"{col}__ordered"
                xtr[ordered_col] = pd.to_numeric(
                    xtr[col].str.extract(r"^ord_(-?\d+(?:\.\d+)?)$", expand=False), errors="coerce"
                )
                xte[ordered_col] = pd.to_numeric(
                    xte[col].str.extract(r"^ord_(-?\d+(?:\.\d+)?)$", expand=False), errors="coerce"
                )
                num_cols.append(ordered_col)
        else:
            xtr[col] = pd.to_numeric(xtr[col], errors="coerce")
            xte[col] = pd.to_numeric(xte[col], errors="coerce")
            num_cols.append(col)
            # Low-cardinality integer/count features can have either ordered or nominal effects.
            finite = combined.dropna()
            integer_like = len(finite) and np.allclose(finite.astype(float), np.round(finite.astype(float)))
            if integer_like and combined.nunique(dropna=True) <= 20:
                cat_view = f"{col}__categorical"
                xtr[cat_view] = xtr[col].astype("Int64").astype("string").fillna("__MISSING__")
                xte[cat_view] = xte[col].astype("Int64").astype("string").fillna("__MISSING__")
                cat_cols.append(cat_view)
    return xtr, xte, cat_cols, num_cols


def sklearn_models(cat_cols, num_cols, n_rows, fallback=False):
    ordinal = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ], remainder="drop")
    trees = 500 if n_rows < 20000 else 350
    result = {
        "extra_trees": Pipeline([
            ("prep", ordinal),
            ("model", ExtraTreesClassifier(
                n_estimators=trees, min_samples_leaf=max(1, int(np.sqrt(n_rows) / 35)),
                max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=SEED,
            )),
        ])
    }
    if fallback:
        result["random_forest"] = Pipeline([
            ("prep", clone(ordinal)),
            ("model", RandomForestClassifier(
                n_estimators=trees, min_samples_leaf=max(2, int(np.sqrt(n_rows) / 25)),
                max_features=0.7, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED + 1,
            )),
        ])
    if n_rows <= 30000:
        onehot = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols),
        ])
        result["logistic"] = Pipeline([
            ("prep", onehot),
            ("model", LogisticRegression(C=0.35, max_iter=800, class_weight="balanced", n_jobs=-1)),
        ])
    return result


def add_boosters(models, cat_cols, n_rows, fast):
    try:
        from catboost import CatBoostClassifier
        iterations = 450 if fast else (750 if n_rows < 25000 else 550)
        models["catboost_d6"] = CatBoostClassifier(
            iterations=iterations, depth=6, learning_rate=0.055, loss_function="Logloss",
            eval_metric="AUC", l2_leaf_reg=5, random_seed=SEED, verbose=False,
            allow_writing_files=False, thread_count=-1,
        )
        if not fast:
            models["catboost_d8"] = CatBoostClassifier(
                iterations=max(500, iterations - 100), depth=8, learning_rate=0.04,
                loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=8,
                random_seed=SEED + 11, verbose=False, allow_writing_files=False, thread_count=-1,
            )
    except Exception as exc:
        print(f"INFO CatBoost unavailable: {exc}")
    try:
        from lightgbm import LGBMClassifier
        leaves = 15 if n_rows < 2000 else 31
        models["lightgbm"] = LGBMClassifier(
            n_estimators=450 if fast else 750, learning_rate=0.035,
            num_leaves=leaves, max_depth=-1, min_child_samples=max(12, int(np.sqrt(n_rows))),
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=2.0,
            random_state=SEED + 23, n_jobs=-1, verbosity=-1,
        )
    except Exception as exc:
        print(f"INFO LightGBM unavailable: {exc}")


def encoded_for_lgbm(xtr, xte, cat_cols):
    a = xtr.copy()
    b = xte.copy()
    for col in cat_cols:
        categories = pd.Index(pd.concat([a[col], b[col]], ignore_index=True).astype(str).unique())
        mapping = pd.Series(np.arange(len(categories)), index=categories)
        a[col] = a[col].astype(str).map(mapping).astype("int32")
        b[col] = b[col].astype(str).map(mapping).astype("int32")
    return a, b


def fit_predict_model(name, model, xtr, xte, y, folds, cat_cols):
    oof = np.zeros(len(xtr), dtype=float)
    pred = np.zeros(len(xte), dtype=float)
    fold_scores = []
    is_catboost = name.startswith("catboost")
    is_lgbm = name == "lightgbm"
    if is_lgbm:
        xtr_use, xte_use = encoded_for_lgbm(xtr, xte, cat_cols)
    else:
        xtr_use, xte_use = xtr, xte
    for fold, (itr, iva) in enumerate(folds):
        fitted = clone(model)
        fit_kwargs = {}
        if is_catboost:
            fit_kwargs = {"cat_features": cat_cols, "eval_set": (xtr_use.iloc[iva], y[iva]),
                          "early_stopping_rounds": 80, "verbose": False}
        elif is_lgbm:
            fit_kwargs = {"categorical_feature": cat_cols}
        fitted.fit(xtr_use.iloc[itr], y[itr], **fit_kwargs)
        oof[iva] = fitted.predict_proba(xtr_use.iloc[iva])[:, 1]
        pred += fitted.predict_proba(xte_use)[:, 1] / len(folds)
        fold_scores.append(roc_auc_score(y[iva], oof[iva]))
    return oof, pred, fold_scores


def greedy_blend(oofs, preds, y, ordered_names):
    best = ordered_names[0]
    blend_oof = rank01(oofs[best])
    blend_pred = rank01(preds[best])
    members = [best]
    best_score = roc_auc_score(y, blend_oof)
    for name in ordered_names[1:]:
        candidate_oof = 0.75 * blend_oof + 0.25 * rank01(oofs[name])
        score = roc_auc_score(y, candidate_oof)
        if score >= best_score - 0.0003:
            blend_oof = candidate_oof
            blend_pred = 0.75 * blend_pred + 0.25 * rank01(preds[name])
            members.append(name)
            best_score = max(best_score, score)
    return blend_oof, blend_pred, members, roc_auc_score(y, blend_oof)


def save_submission(sample, target, pred, filename):
    out = sample.copy()
    out[target] = np.clip(pred, 1e-7, 1 - 1e-7)
    out.to_csv(filename, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    started = time.time()
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")
    target, id_col, features = find_columns(train, test, sample)
    y, mapping = normalize_target(train[target])
    xtr, xte, cat_cols, num_cols = prepare_frames(train, test, features)
    n_splits = 3 if (args.fast or len(train) > 30000) else 4
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(xtr, y))
    models = sklearn_models(cat_cols, num_cols, len(train), fallback=args.fallback)
    if not args.fallback:
        add_boosters(models, cat_cols, len(train), args.fast)
    print(json.dumps({"rows": len(train), "test_rows": len(test), "features": len(features),
                      "categorical": len(cat_cols), "numeric": len(num_cols), "folds": n_splits,
                      "models": list(models)}, sort_keys=True))
    oofs, preds, results = {}, {}, []
    for name, model in models.items():
        try:
            t0 = time.time()
            oof, pred, fold_scores = fit_predict_model(name, model, xtr, xte, y, folds, cat_cols)
            score = roc_auc_score(y, oof)
            oofs[name], preds[name] = oof, pred
            results.append({"name": name, "cv_auc": score, "fold_auc": fold_scores,
                            "seconds": round(time.time() - t0, 1)})
            print(f"MODEL {name} cv_auc={score:.6f} folds={','.join(f'{s:.5f}' for s in fold_scores)}")
        except Exception as exc:
            print(f"MODEL_FAILED {name}: {type(exc).__name__}: {exc}")
    if not results:
        raise RuntimeError("All models failed")
    results.sort(key=lambda r: r["cv_auc"], reverse=True)
    names = [r["name"] for r in results]
    _, blend_pred, members, blend_score = greedy_blend(oofs, preds, y, names)
    candidates = [("blend", blend_pred, blend_score, members)]
    for item in results:
        candidates.append((item["name"], rank01(preds[item["name"]]), item["cv_auc"], [item["name"]]))
    # A stable broad average is useful when CV is noisy on tiny datasets.
    top = names[: min(3, len(names))]
    broad = np.mean([rank01(preds[n]) for n in top], axis=0)
    broad_oof = np.mean([rank01(oofs[n]) for n in top], axis=0)
    candidates.append(("broad_blend", broad, roc_auc_score(y, broad_oof), top))
    if len(names) >= 2:
        top2 = names[:2]
        pair = np.mean([rank01(preds[n]) for n in top2], axis=0)
        pair_oof = np.mean([rank01(oofs[n]) for n in top2], axis=0)
        candidates.append(("top2_blend", pair, roc_auc_score(y, pair_oof), top2))
    candidates.sort(key=lambda x: x[2], reverse=True)
    files, seen = [], []
    for idx, (name, pred, score, members) in enumerate(candidates):
        if any(np.corrcoef(pred, p)[0, 1] > 0.99998 for p in seen):
            continue
        filename = f"candidate_{len(files)+1:02d}_{name}.csv"
        save_submission(sample, target, pred, filename)
        diversity = 1.0 if not seen else float(1 - max(np.corrcoef(pred, p)[0, 1] for p in seen))
        files.append({"file": filename, "name": name, "cv_auc": score,
                      "members": members, "diversity_from_earlier": diversity})
        seen.append(pred)
        if len(files) >= 10:
            break
    manifest = {
        "schema": {"target": target, "id": id_col, "features": len(features),
                   "categorical": cat_cols, "numeric": num_cols, "target_mapping": {str(k): v for k, v in mapping.items()}},
        "models": results, "candidates": files,
        "recommended_diverse_file": files[1]["file"] if len(files) > 1 else files[0]["file"],
        "elapsed_seconds": round(time.time() - started, 1), "seed": SEED,
    }
    Path("automl_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("CANDIDATES " + " ".join(item["file"] for item in files))
    print(f"DONE elapsed_seconds={manifest['elapsed_seconds']}")


if __name__ == "__main__":
    main()
