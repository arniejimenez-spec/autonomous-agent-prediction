---
name: tabular-automl
description: Runs a two-stage family-diverse model portfolio and a bounded public-led refinement frontier for binary tabular classification.
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
- runs spline-additive, histogram-threshold, and small-data RBF probes to distinguish synthetic DGP archetypes using train-only out-of-fold evidence;
- adds smoother depth-4 and ordered-boosting CatBoost variants on small datasets, plus two-seed averages when a small dataset is entirely numeric;
- creates leakage-safe out-of-fold predictions;
- builds robust rank ensembles, including a conservatively weighted top-two blend, without using test labels;
- preserves the complete v6 ensemble family whenever a later specialist is enabled;
- audits learned routing offline with entire datasets held out, falling back to the stronger highest-CV hedge when the learned selector does not clear that benchmark;
- writes at most thirteen compact `p01.csv`, `p02.csv`, ... files matching `sample_submission.csv` exactly, with p01 reserved for the strongest train-only CV hedge;
- prioritizes direct representatives from CatBoost, LightGBM, forests, histogram boosting, linear/additive models, kernels, and routed specialists instead of filling Stage 1 with near-duplicate ensembles;
- appends shallow/ordered CatBoost and specialist-excluding baseline safety files without displacing the first ten family-diverse candidates;
- writes `automl_manifest.json` with CV scores, file order, diversity, and recommendations.

After submitting every Stage 1 file, run `scripts/refine.py` once with `--first`, `--second`, and `--third` set to the three public-leading Stage 1 filenames and `--hedge` set to p01. The script creates eight bounded rank blends named `r01.csv` through `r08.csv`. Public feedback chooses only the blend inputs; no labels are available to either script.

Use `--fast` only after a normal Stage 1 run fails or the remaining runtime is under 20 minutes. Use `--fallback` only if optional boosting libraries fail.

The scripts intentionally print no diagnostics. Stage 1 prints compact `CV_HEDGE`, `STAGE1`, and `CANDIDATES` lines plus `DONE`; Stage 2 prints only `REFINED`. Submit every printed candidate. Final selection pairs the CV hedge with the Stage 1 public leader, allowing a refined file to replace that leader only after a public gain of at least 0.0001.
