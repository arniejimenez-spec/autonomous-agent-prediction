---
name: tabular-automl
description: Runs a pre-tested, budget-aware model portfolio for mixed-type binary tabular classification and produces ranked submission candidates.
---

# Tabular AutoML

Use this skill exactly once at the beginning of a binary classification task.

## Script

Run `scripts/automl.py` in the sandbox working directory. It automatically:

- infers the target and identifier from the supplied CSV files;
- handles numerical, categorical, ordinal, and missing values, preserving both ordered and categorical views when appropriate;
- cross-validates CatBoost, LightGBM, ExtraTrees, and regularized linear models;
- creates leakage-safe out-of-fold predictions;
- builds robust rank ensembles without using test labels;
- writes `candidate_*.csv` files matching `sample_submission.csv` exactly;
- writes `automl_manifest.json` with CV scores, file order, diversity, and recommendations.

Use `--fast` only after a normal run fails or the remaining runtime is under 20 minutes. Use `--fallback` only if optional boosting libraries fail.

Submit at most the first eight files listed in the manifest. Public leaderboard feedback is for coarse model selection only, never prediction-level tuning.
