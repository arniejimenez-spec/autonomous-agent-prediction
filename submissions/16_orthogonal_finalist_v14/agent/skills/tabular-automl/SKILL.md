---
name: tabular-automl
description: Runs a family-diverse binary tabular portfolio and identifies direct-model finalists by rank disagreement with a train-only CV hedge.
---

# Tabular AutoML

Use this skill exactly once at the beginning of a binary classification task.

## Script

Run `scripts/automl.py` using `run_skill_script(skill_name="tabular-automl", file_path="scripts/automl.py")`. ADK materializes skills in a temporary directory; the script automatically switches to the harness's persistent `/work` directory before reading or writing competition files.

The script:

- infers the target, identifier, label mapping, numeric columns, and categorical columns;
- fits leakage-safe cross-validated CatBoost, LightGBM, ExtraTrees, regularized linear, histogram, Random Forest, XGBoost, spline, quadratic, target-encoding, and small-data kernel models when their established routes apply;
- creates the historical rank-ensemble frontier;
- reserves p01 for the strongest train-only CV hedge;
- writes at most thirteen compact `pNN.csv` files matching `sample_submission.csv` exactly;
- appends shallow/ordered CatBoost and specialist-excluding safety predictions when available;
- measures every direct model's rank disagreement with p01 without using test labels;
- writes `automl_manifest.json` with CV scores, model families, and hedge diversity.

Use `--fast` only after a normal run fails or the remaining runtime is under 20 minutes. Use `--fallback` only if optional boosting libraries fail.

The script prints only compact `CV_HEDGE`, `ORTHOGONAL`, and `CANDIDATES` lines plus `DONE`. Submit every candidate. Pair p01 with the most diverse direct family whose public score is within 0.005 AUC of the best direct-model public score. Do not generate public-tuned blends.
