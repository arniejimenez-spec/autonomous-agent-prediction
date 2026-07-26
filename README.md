# Autonomous Agent Prediction — Fingerprint-Routed Tabular Agent

This repository contains an Agent Config for Kaggle's Autonomous Agent Prediction beta competition. During each evaluation session, the agent must solve two unseen binary-classification mini-competitions, submit candidate predictions through the competition tools, and select exactly two finalists.

## Current status

| Item | Status |
|---|---|
| Best official Kaggle score | **0.822 AUC** |
| Versions at the official best | v3, v4, v5, and v6 |
| Current candidate | **v7 — validated locally, official score pending** |
| Agent source | [`submissions/08_fingerprint_routed_automl_v7/agent/`](submissions/08_fingerprint_routed_automl_v7/agent/) |
| Kaggle build notebook | [`notebooks/build_agent_submission_v7.ipynb`](notebooks/build_agent_submission_v7.ipynb) |
| Design notes | [`docs/V7.md`](docs/V7.md) |

The official score improved from 0.781 to 0.814 and then to 0.822. Versions v4–v6 added useful local safeguards and model diversity but did not move the black-box score beyond 0.822. V7 is the next submission candidate; it should not be described as an official improvement until Kaggle returns a score.

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
| v7 | Pending | Adds dataset fingerprinting, routed specialists, and a deterministic baseline hedge |

The 16-task replay is a development signal, not a substitute for the hidden Kaggle evaluation. V7 raises selected-private replay AUC from **0.805450** for v6 to **0.805604**, with no regression on the tasks where its new routes activate. On the normal-mode `train_15` check, the selected v7 pair scores **0.868734** versus **0.866506** for the safe baseline.

## V7 strategy

The LLM acts as a lightweight orchestrator while a deterministic, pre-tested skill performs the modeling:

1. immediately submit the sample prediction as a fail-safe;
2. infer the target, ID, binary-label mapping, feature types, and dataset fingerprint;
3. route the dataset to a bounded portfolio of CatBoost, LightGBM, ExtraTrees, logistic regression, Random Forest, XGBoost, and specialized feature pipelines;
4. produce leakage-safe out-of-fold predictions and rank-based blends;
5. keep one train-only-selected safe hedge while the other slot explores the strongest distinct public-frontier candidate;
6. call `select_submissions` with exactly two valid predictions.

Specialists are activated conservatively:

- shallow and ordered CatBoost variants for small-to-medium categorical tasks;
- cross-fitted target encoding only for tiny datasets with many categorical views;
- Random Forest for selected medium-sized tasks;
- one-hot XGBoost for sufficiently large, categorical datasets;
- quadratic interactions only for numeric-dominant geometry.

The agent uses deterministic seeds, caps the number of candidates below the competition limit, avoids network access and runtime installation, and falls back cleanly when optional boosting libraries are unavailable.

## Build the current submission

Python 3.13 is recommended because the evaluator dependency chain does not currently build cleanly on Python 3.14.

```powershell
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe catboost lightgbm xgboost
```

Validate and package v7:

```powershell
.\.venv\Scripts\python.exe validate_submission.py `
  --agent-dir submissions/08_fingerprint_routed_automl_v7/agent

.\.venv\Scripts\python.exe scripts/package_submission.py `
  --experiment 08_fingerprint_routed_automl_v7
```

The generated file is:

```text
submissions/08_fingerprint_routed_automl_v7/submission.zip
```

Its SHA-256 for the currently validated build is:

```text
8EEA61B64A595A483CC54ADC633F4676106A0A3215B1BA17A1CC9A736337898D
```

The ZIP is intentionally excluded from Git and can be rebuilt by the notebook, locally, or in CI. It must contain `agent.yaml` at its root; do not zip the parent `agent/` directory.

## Build in a Kaggle notebook

Upload and run [`notebooks/build_agent_submission_v7.ipynb`](notebooks/build_agent_submission_v7.ipynb). It is self-contained and writes:

```text
/kaggle/working/submission.zip
```

Download that ZIP from the notebook output and upload it as the competition submission.

After changing the agent source, regenerate the notebook with:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py `
  --experiment 08_fingerprint_routed_automl_v7 `
  --output notebooks/build_agent_submission_v7.ipynb
```

## Local evaluation

Quick deterministic meta-evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/meta_evaluate.py `
  train_13 train_06 train_16 train_11 `
  --fast `
  --experiment 08_fingerprint_routed_automl_v7
```

Run all sixteen solved tasks:

```powershell
.\.venv\Scripts\python.exe scripts/meta_evaluate.py `
  --fast `
  --experiment 08_fingerprint_routed_automl_v7
```

Replay the v7 routing and selection policy:

```powershell
.\.venv\Scripts\python.exe scripts/replay_v7_policy.py `
  submissions/08_fingerprint_routed_automl_v7/meta_results_all.json `
  --hedges submissions/08_fingerprint_routed_automl_v7/policy_hedges.json
```

For full agent evaluation, copy `.env.example` to `.env`, configure the API key for the model in `agent.yaml`, and install Docker or Podman:

```powershell
.\.venv\Scripts\python.exe run_local_eval.py `
  --submission-dir submissions/08_fingerprint_routed_automl_v7/agent `
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
  08_fingerprint_routed_automl_v7/  current candidate
notebooks/                           self-contained Kaggle builders
scripts/                             packaging, evaluation, and replay tools
docs/                                version-specific experiment notes
kaggle-kaggle-skill/                 organizer documentation
```

Detailed iteration notes are available in [`docs/V2.md`](docs/V2.md), [`docs/V3.md`](docs/V3.md), [`docs/V4.md`](docs/V4.md), [`docs/V5.md`](docs/V5.md), [`docs/V6.md`](docs/V6.md), and [`docs/V7.md`](docs/V7.md).

The organizer-supplied `sample_submission/` remains unchanged.

## License

Code authored in this repository is released under the MIT License. Competition datasets and organizer starter materials retain their original CC BY 4.0 terms.
