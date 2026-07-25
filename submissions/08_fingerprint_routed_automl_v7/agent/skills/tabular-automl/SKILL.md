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
- cross-validates CatBoost, LightGBM, ExtraTrees, regularized linear models, and a quadratic interaction model on suitable numeric-dominant tasks;
- routes XGBoost and Random Forest diversity candidates only to dataset archetypes supported by meta-evaluation evidence;
- fingerprints dataset size and feature-type geometry to route shallow/ordered and cross-fitted target-encoding specialists;
- adds smoother depth-4 and ordered-boosting CatBoost variants on small datasets, plus two-seed averages when a small dataset is entirely numeric;
- creates leakage-safe out-of-fold predictions;
- builds robust rank ensembles, including a conservatively weighted top-two blend, without using test labels;
- preserves the complete v6 ensemble family whenever a v7 specialist is enabled;
- marks exact historical v5/v6-equivalent candidates and chooses a hedge family from train-only fingerprints and pilot CV;
- writes `candidate_*.csv` files matching `sample_submission.csv` exactly;
- writes `automl_manifest.json` with CV scores, file order, diversity, and recommendations.

Use `--fast` only after a normal run fails or the remaining runtime is under 20 minutes. Use `--fallback` only if optional boosting libraries fail.

Submit at most the first sixteen files listed in the manifest. If `SAFE_HEDGE` is printed, pair that exact file with the highest public scorer, using the second-highest public scorer only when the hedge itself leads. Otherwise select the two highest public scorers. Public feedback must never be used to generate or alter predictions.
