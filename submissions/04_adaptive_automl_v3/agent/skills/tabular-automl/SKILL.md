---
name: tabular-automl
description: Runs a pre-tested, budget-aware model portfolio for mixed-type binary tabular classification and produces ranked submission candidates.
---

# Tabular AutoML

Use this skill exactly once at the beginning of a binary classification task.

## Script

Run `scripts/automl.py` using `run_skill_script(skill_name="tabular-automl", file_path="scripts/automl.py")`. ADK materializes skills in a temporary directory; the script automatically switches to the harness's persistent `/work` directory before reading or writing competition files. It then:

- infers the target and identifier from the supplied CSV files;
- handles numerical, categorical, ordinal, and missing values, preserving both ordered and categorical views when appropriate;
- cross-validates CatBoost, LightGBM, ExtraTrees, and regularized linear models;
- adds smoother depth-4 and ordered-boosting CatBoost variants when the training set is small;
- creates leakage-safe out-of-fold predictions;
- builds robust rank ensembles, including a conservatively weighted top-two blend, without using test labels;
- writes `candidate_*.csv` files matching `sample_submission.csv` exactly;
- writes `automl_manifest.json` with CV scores, file order, diversity, and recommendations.

Use `--fast` only after a normal run fails or the remaining runtime is under 20 minutes. Use `--fallback` only if optional boosting libraries fail.

Submit at most the first twelve files listed in the manifest. Public leaderboard feedback is for coarse model selection only, never prediction-level tuning.
