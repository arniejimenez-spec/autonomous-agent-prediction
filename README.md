# Autonomous Agent Prediction — DGP-Probing Tabular Agent

This repository contains an Agent Config for Kaggle's Autonomous Agent Prediction beta competition. During each evaluation session, the agent must solve two unseen binary-classification mini-competitions, submit candidate predictions through the competition tools, and select exactly two finalists.

## Current status

| Item | Status |
|---|---|
| Best official Kaggle score | **0.822 AUC** |
| Versions at the official best | v3, v4, v5, v6, v7, and v8.1 |
| Current candidate | **v9 — conservative full-data refit bagging, official score pending** |
| Agent source | [`submissions/11_full_refit_bagging_v9/agent/`](submissions/11_full_refit_bagging_v9/agent/) |
| Kaggle build notebook | [`notebooks/build_agent_submission_v9.ipynb`](notebooks/build_agent_submission_v9.ipynb) |
| Design notes | [`docs/V9.md`](docs/V9.md) |

The official score improved from 0.781 to 0.814 and then to 0.822. Versions v4–v7 remained at 0.822. V8 returned 0.500, indicating an execution-path failure that left the chance-level fallback in place. V8.1 restored the compact execution path and returned to 0.822. V9 preserves that runtime-safe path while adding two tightly bounded full-training candidates.

## Score history

| Version | Official score | What changed |
|---|---:|---|
| v1 | 0.781 | Initial robust tabular AutoML agent |
| v2 | Invalid run | Agent completed without calling `submit_predictions` |
| v2.1 | 0.814 | Guaranteed an early valid submission and fixed skill execution from the persistent work directory |
| v3 | **0.822** | Added small-data CatBoost specialists, weighted blends, and explicit safe fallbacks |
| v4 | **0.822** | Aligned final selection with the best-private-of-two evaluation and routed numeric seed averaging |
| v5 | **0.822** | Added regularized quadratic interactions for numeric-dominant datasets |
| v6 | **0.822** | Added carefully routed XGBoost and Random Forest diversity |
| v7 | **0.822** | Added dataset fingerprinting, routed specialists, and a deterministic baseline hedge |
| v8 | 0.500 | Likely runtime failure; official output remained at chance level |
| v8.1 | **0.822** | Caps the frontier at ten, emits a 135-character plan, and selects immediately |
| v9 | Pending | Adds one repeated-CV bag and one conservative full-data refit blend without changing the historical CV hedge |

The 16-task replay is a development signal, not a substitute for the hidden Kaggle evaluation. V9 raises selected-private replay AUC from **0.805725** for v8.1 to **0.805892**, a **+0.000167** mean improvement with zero regressions. A broader refit frontier was tested and rejected because it crowded proven candidates out of the ten-file cap.

## V9 strategy

The LLM acts as a lightweight orchestrator while a deterministic, pre-tested skill performs the modeling:

1. immediately submit the sample prediction as a fail-safe;
2. infer the target, ID, binary-label mapping, feature types, and dataset fingerprint;
3. route the dataset to a bounded portfolio of CatBoost, LightGBM, ExtraTrees, logistic regression, Random Forest, XGBoost, and DGP probes;
4. produce leakage-safe out-of-fold predictions and rank-based blends;
5. on datasets with at least 1,000 rows, add one second-seed CV bag and blend one full-data refit into the proven historical hedge at 20%;
6. keep the best historical train-CV candidate as one hedge while the other slot explores the strongest public-frontier candidate;
7. call `select_submission` with exactly two valid predictions.

Specialists are activated conservatively:

- shallow and ordered CatBoost variants for small-to-medium categorical tasks;
- cross-fitted target encoding only for tiny datasets with many categorical views;
- Random Forest for selected medium-sized tasks;
- one-hot XGBoost for sufficiently large, categorical datasets;
- quadratic interactions only for numeric-dominant geometry.
- spline-additive logistic regression for smooth effects;
- histogram gradient boosting for threshold-heavy rules;
- RBF SVC only on small, low-dimensional tasks.

The agent submits at most ten modeled candidates after its initial fallback. Compact `p01.csv` filenames keep the complete plan visible even under a 500-character output cap, and final selection occurs immediately after submission.

## Build the current submission

Python 3.13 is recommended because the evaluator dependency chain does not currently build cleanly on Python 3.14.

```powershell
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe catboost lightgbm xgboost
```

Validate and package v9:

```powershell
.\.venv\Scripts\python.exe validate_submission.py `
  --agent-dir submissions/11_full_refit_bagging_v9/agent

.\.venv\Scripts\python.exe scripts/package_submission.py `
  --experiment 11_full_refit_bagging_v9
```

The generated file is:

```text
submissions/11_full_refit_bagging_v9/submission.zip
```

The validated v9 build SHA-256 is recorded in [`docs/V9.md`](docs/V9.md).

The ZIP is intentionally excluded from Git and can be rebuilt by the notebook, locally, or in CI. It must contain `agent.yaml` at its root; do not zip the parent `agent/` directory.

## Build in a Kaggle notebook

Upload and run [`notebooks/build_agent_submission_v9.ipynb`](notebooks/build_agent_submission_v9.ipynb). It is self-contained and writes:

```text
/kaggle/working/submission.zip
```

Download that ZIP from the notebook output and upload it as the competition submission.

After changing the agent source, regenerate the notebook with:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py `
  --experiment 11_full_refit_bagging_v9 `
  --output notebooks/build_agent_submission_v9.ipynb
```

## Local evaluation

Quick deterministic meta-evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/meta_evaluate.py `
  train_13 train_06 train_16 train_11 `
  --fast `
  --experiment 11_full_refit_bagging_v9
```

Run all sixteen solved tasks:

```powershell
.\.venv\Scripts\python.exe scripts/meta_evaluate.py `
  --fast `
  --experiment 11_full_refit_bagging_v9
```

Compare v9's exact two-selection policy with v8.1:

```powershell
.\.venv\Scripts\python.exe scripts/replay_v9_policy.py `
  submissions/11_full_refit_bagging_v9/meta_results_all.json `
  --baseline-results submissions/09_meta_routed_automl_v8/meta_results_all.json
```

Compare the underlying v8 policy against v7:

```powershell
.\.venv\Scripts\python.exe scripts/replay_v8_policy.py `
  submissions/09_meta_routed_automl_v8/meta_results_all.json `
  --baseline-results submissions/08_fingerprint_routed_automl_v7/meta_results_all.json `
  --baseline-hedges submissions/08_fingerprint_routed_automl_v7/policy_hedges.json
```

For full agent evaluation, copy `.env.example` to `.env`, configure the API key for the model in `agent.yaml`, and install Docker or Podman:

```powershell
.\.venv\Scripts\python.exe run_local_eval.py `
  --submission-dir submissions/11_full_refit_bagging_v9/agent `
  --dataset train_01 `
  --metric roc_auc
```

The supplied `solution.csv` files are used only by repository-level development scripts. They are never bundled into the Agent Config.

## Repository map

```text
submissions/
  01_robust_automl/                 v1
  02_order_aware_automl/            v2
  03_fail_safe_automl/              v2.1
  04_adaptive_automl_v3/            v3
  05_selection_stable_automl_v4/    v4
  06_interaction_routed_automl_v5/  v5
  07_tree_diversity_automl_v6/      v6
  08_fingerprint_routed_automl_v7/  v7
  09_meta_routed_automl_v8/         v8 runtime-failed experiment
  10_runtime_safe_dgp_v81/          v8.1 runtime-safe official 0.822
  11_full_refit_bagging_v9/         conservative full-refit current candidate
notebooks/                           self-contained Kaggle builders
scripts/                             packaging, evaluation, and replay tools
docs/                                version-specific experiment notes
kaggle-kaggle-skill/                 organizer documentation
```

Detailed iteration notes are available in [`docs/V2.md`](docs/V2.md), [`docs/V3.md`](docs/V3.md), [`docs/V4.md`](docs/V4.md), [`docs/V5.md`](docs/V5.md), [`docs/V6.md`](docs/V6.md), [`docs/V7.md`](docs/V7.md), [`docs/V8.md`](docs/V8.md), [`docs/V8.1.md`](docs/V8.1.md), and [`docs/V9.md`](docs/V9.md).

The organizer-supplied `sample_submission/` remains unchanged.

## License

Code authored in this repository is released under the MIT License. Competition datasets and organizer starter materials retain their original CC BY 4.0 terms.
