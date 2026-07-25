#!/usr/bin/env python3
"""Budget-aware mixed-type AutoML for the Kaggle-in-Kaggle sandbox."""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
SEED = 20260717


def enter_competition_workdir() -> Path:
    """Use the persistent harness directory, not ADK's temporary skill folder."""
    configured = os.environ.get("KAGGLE_WORK_DIR")
    candidates = [Path.cwd()]
    if configured:
        candidates.append(Path(configured))
    candidates.extend([Path("/work"), Path("/kaggle/working")])
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("train.csv", "test.csv", "sample_submission.csv")):
            os.chdir(candidate)
            return candidate
    raise FileNotFoundError(
        "Competition CSVs were not found in the current directory, /work, or /kaggle/working"
    )


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
        if 8 <= len(num_cols) <= 30 and len(cat_cols) <= 4:
            quadratic = ColumnTransformer([
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("interactions", PolynomialFeatures(degree=2, include_bias=False)),
                    ("rescale", StandardScaler()),
                ]), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols),
            ], remainder="drop")
            result["quadratic_logistic"] = Pipeline([
                ("prep", quadratic),
                ("model", LogisticRegression(
                    C=0.05, max_iter=1200, class_weight="balanced", n_jobs=-1,
                )),
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
        if n_rows < 4000:
            small_iterations = 400 if fast else 650
            models["catboost_d4_smooth"] = CatBoostClassifier(
                iterations=small_iterations, depth=4, learning_rate=0.045,
                loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=10,
                random_strength=1.5, random_seed=SEED + 5, verbose=False,
                allow_writing_files=False, thread_count=-1,
            )
            models["catboost_ordered_d5"] = CatBoostClassifier(
                iterations=small_iterations, depth=5, learning_rate=0.045,
                boosting_type="Ordered", loss_function="Logloss", eval_metric="AUC",
                l2_leaf_reg=8, random_strength=0.8, random_seed=SEED + 7,
                verbose=False, allow_writing_files=False, thread_count=-1,
            )
            # Seed averaging pays for itself on small, entirely numeric tasks.
            # Mixed categorical tasks already get diversity from representation
            # and model-family blends, while duplicate CatBoost seeds add cost.
            if not cat_cols:
                models["catboost_d4_smooth_seed_b"] = CatBoostClassifier(
                    iterations=small_iterations, depth=4, learning_rate=0.045,
                    loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=10,
                    random_strength=1.5, random_seed=SEED + 105, verbose=False,
                    allow_writing_files=False, thread_count=-1,
                )
                models["catboost_ordered_d5_seed_b"] = CatBoostClassifier(
                    iterations=small_iterations, depth=5, learning_rate=0.045,
                    boosting_type="Ordered", loss_function="Logloss", eval_metric="AUC",
                    l2_leaf_reg=8, random_strength=0.8, random_seed=SEED + 107,
                    verbose=False, allow_writing_files=False, thread_count=-1,
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


def weighted_top2_blend(oofs, preds, y, ordered_names):
    """Tune only one coarse weight to limit blend-selection overfitting."""
    first, second = ordered_names[:2]
    r1_oof, r2_oof = rank01(oofs[first]), rank01(oofs[second])
    r1_pred, r2_pred = rank01(preds[first]), rank01(preds[second])
    weights = [0.5] if len(y) < 1500 else [0.35, 0.5, 0.65, 0.8]
    scored = []
    for weight in weights:
        blended = weight * r1_oof + (1.0 - weight) * r2_oof
        scored.append((roc_auc_score(y, blended), weight))
    score, weight = max(scored)
    pred = weight * r1_pred + (1.0 - weight) * r2_pred
    return pred, score, [first, second], weight


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
    workdir = enter_competition_workdir()
    print(f"WORKDIR {workdir}")
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
        weighted, weighted_score, weighted_members, weight = weighted_top2_blend(oofs, preds, y, names)
        candidates.append((f"weighted_top2_{weight:.2f}", weighted, weighted_score, weighted_members))
    for ensemble_name, first, second in (
        ("catboost_d4_seed_average", "catboost_d4_smooth", "catboost_d4_smooth_seed_b"),
        ("catboost_ordered_d5_seed_average", "catboost_ordered_d5", "catboost_ordered_d5_seed_b"),
    ):
        if first in oofs and second in oofs:
            averaged_oof = 0.5 * rank01(oofs[first]) + 0.5 * rank01(oofs[second])
            averaged_pred = 0.5 * rank01(preds[first]) + 0.5 * rank01(preds[second])
            candidates.append((
                ensemble_name, averaged_pred, roc_auc_score(y, averaged_oof), [first, second],
            ))
    # Preserve the complete v2.1 ensemble family so adaptive models can never
    # displace the proven baseline combinations on a small, noisy CV split.
    baseline_names = [
        name for name in names
        if name not in {
            "catboost_d4_smooth", "catboost_ordered_d5",
            "catboost_d4_smooth_seed_b", "catboost_ordered_d5_seed_b",
        }
    ]
    if len(baseline_names) >= 2 and baseline_names != names:
        _, baseline_pred, baseline_members, baseline_score = greedy_blend(
            oofs, preds, y, baseline_names
        )
        candidates.append(("v21_blend", baseline_pred, baseline_score, baseline_members))
        baseline_top2 = baseline_names[:2]
        baseline_pair = np.mean([rank01(preds[n]) for n in baseline_top2], axis=0)
        baseline_pair_oof = np.mean([rank01(oofs[n]) for n in baseline_top2], axis=0)
        candidates.append((
            "v21_top2_blend", baseline_pair,
            roc_auc_score(y, baseline_pair_oof), baseline_top2,
        ))
        baseline_top3 = baseline_names[: min(3, len(baseline_names))]
        baseline_broad = np.mean([rank01(preds[n]) for n in baseline_top3], axis=0)
        baseline_broad_oof = np.mean([rank01(oofs[n]) for n in baseline_top3], axis=0)
        candidates.append((
            "v21_broad_blend", baseline_broad,
            roc_auc_score(y, baseline_broad_oof), baseline_top3,
        ))
    # Preserve the exact v3 model family so new seed variants cannot displace
    # the previously validated adaptive ensembles.
    v3_names = [name for name in names if not name.endswith("_seed_b")]
    if len(v3_names) >= 2 and v3_names != names:
        _, v3_pred, v3_members, v3_score = greedy_blend(oofs, preds, y, v3_names)
        candidates.append(("v3_blend", v3_pred, v3_score, v3_members))
        v3_top2 = v3_names[:2]
        v3_pair = np.mean([rank01(preds[n]) for n in v3_top2], axis=0)
        v3_pair_oof = np.mean([rank01(oofs[n]) for n in v3_top2], axis=0)
        candidates.append((
            "v3_top2_blend", v3_pair, roc_auc_score(y, v3_pair_oof), v3_top2,
        ))
        v3_top3 = v3_names[: min(3, len(v3_names))]
        v3_broad = np.mean([rank01(preds[n]) for n in v3_top3], axis=0)
        v3_broad_oof = np.mean([rank01(oofs[n]) for n in v3_top3], axis=0)
        candidates.append((
            "v3_broad_blend", v3_broad, roc_auc_score(y, v3_broad_oof), v3_top3,
        ))
    # Preserve the exact v4 family whenever the experimental interaction
    # model is present, preventing it from displacing validated ensembles.
    v4_names = [name for name in names if name != "quadratic_logistic"]
    if len(v4_names) >= 2 and v4_names != names:
        _, v4_pred, v4_members, v4_score = greedy_blend(oofs, preds, y, v4_names)
        candidates.append(("v4_blend", v4_pred, v4_score, v4_members))
        v4_top2 = v4_names[:2]
        v4_pair = np.mean([rank01(preds[n]) for n in v4_top2], axis=0)
        v4_pair_oof = np.mean([rank01(oofs[n]) for n in v4_top2], axis=0)
        candidates.append((
            "v4_top2_blend", v4_pair, roc_auc_score(y, v4_pair_oof), v4_top2,
        ))
        v4_weighted, v4_weighted_score, v4_weighted_members, v4_weight = weighted_top2_blend(
            oofs, preds, y, v4_names
        )
        candidates.append((
            f"v4_weighted_top2_{v4_weight:.2f}", v4_weighted,
            v4_weighted_score, v4_weighted_members,
        ))
        v4_top3 = v4_names[: min(3, len(v4_names))]
        v4_broad = np.mean([rank01(preds[n]) for n in v4_top3], axis=0)
        v4_broad_oof = np.mean([rank01(oofs[n]) for n in v4_top3], axis=0)
        candidates.append((
            "v4_broad_blend", v4_broad, roc_auc_score(y, v4_broad_oof), v4_top3,
        ))
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
        if len(files) >= 16:
            break
    manifest = {
        "schema": {"target": target, "id": id_col, "features": len(features),
                   "categorical": cat_cols, "numeric": num_cols, "target_mapping": {str(k): v for k, v in mapping.items()}},
        "models": results, "candidates": files,
        "selection_policy": "select the two highest public scorers",
        "elapsed_seconds": round(time.time() - started, 1), "seed": SEED,
    }
    Path("automl_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("CANDIDATES " + " ".join(item["file"] for item in files))
    print(f"DONE elapsed_seconds={manifest['elapsed_seconds']}")


if __name__ == "__main__":
    main()
