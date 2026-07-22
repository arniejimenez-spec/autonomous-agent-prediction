# Autonomous Agent Prediction — Robust Tabular AutoML Agent

This repository contains a competition-ready Agent Config for Kaggle's Autonomous Agent Prediction beta competition. The agent solves an unseen binary tabular task inside a 60-minute CPU sandbox, uses public leaderboard feedback conservatively, and selects two final submissions.

## What is included

- `submissions/04_adaptive_automl_v3/agent/` — adaptive v3 with small-data CatBoost variants and baseline-preserving blends
- `notebooks/build_agent_submission_v3.ipynb` — self-contained adaptive v3 Kaggle notebook
- `submissions/05_selection_stable_automl_v4/agent/` — selection-stable v4 with routed numeric seed averaging
- `notebooks/build_agent_submission_v4.ipynb` — self-contained selection-stable v4 Kaggle notebook

- `submissions/01_robust_automl/agent/` — uploadable Agent Config source
- `submissions/02_order_aware_automl/agent/` — controlled order-aware v2 Agent Config
- `submissions/03_fail_safe_automl/agent/` — corrected v2.1 with guaranteed baseline submission and persistent-workdir handling
- `submissions/01_robust_automl/submission.zip` — locally generated Kaggle artifact (excluded from Git; rebuilt by the notebook or CI)
- `notebooks/build_agent_submission.ipynb` — self-contained Kaggle notebook that rebuilds the ZIP
- `notebooks/build_agent_submission_v2.ipynb` — self-contained v2 Kaggle notebook
- `notebooks/build_agent_submission_v21.ipynb` — self-contained corrected v2.1 Kaggle notebook
- `scripts/meta_evaluate.py` — evaluates the deterministic ML skill on the 16 solved tasks
- `scripts/build_notebook.py` — regenerates the self-contained notebook from the agent directory
- `scripts/package_submission.py` — cross-platform Agent Config packager
- `scripts/package_submission.ps1` — packages the Agent Config on Windows
- `kaggle-kaggle-skill/` — organizer-supplied competition documentation

The organizer-supplied `sample_submission/` is intentionally unchanged.

## Current versions

- **v1** scored **0.781 AUC** in the black-box competition evaluation.
- **v2** preserves explicit ordinal ordering alongside categorical representations and adds a top-two rank blend. Across the 16 solved tasks it improves mean best-candidate AUC from 0.80209 to 0.80241 and simulated public-selected private AUC from 0.80285 to 0.80320.
- **v2.1** scored **0.814 AUC** after fixing ADK skill execution from a temporary directory, making a valid baseline submission before modeling, naming exact skill-tool calls, and upgrading the orchestration model for reliable tool use.
- **v3** scored **0.822 AUC**. It adds shallow and ordered CatBoost models on small datasets, a weighted top-two blend on larger datasets, and explicit v2.1 ensemble fallbacks.
- **v4** aligns final selection with the evaluator's best-private-of-two rule and adds two-seed CatBoost averaging only for small all-numeric datasets. Mean best-candidate AUC rises from 0.803754 to 0.803774, while simulated top-two-public private AUC reaches 0.804807, within 0.000010 of the available-candidate oracle. Use v4 for the next submission.

## Architecture

The LLM is a low-cost orchestration layer. Expensive and error-prone modeling work lives in a pre-tested sandbox skill:

1. infer the target, ID, feature types, and binary label mapping;
2. run stratified cross-validation across CatBoost, LightGBM, ExtraTrees, and regularized logistic regression;
3. generate leakage-safe out-of-fold predictions;
4. create greedy and broad rank blends;
5. write ranked candidate CSV files and a machine-readable manifest;
6. submit a capped set of ranked candidates and select two robust, distinct finalists.

This structure spends LLM tokens on orchestration instead of generating ad hoc training code. The current fail-safe agents use `gemini-3.5-flash` for reliable skill calls within the official session budget.

## Local setup

Python 3.13 is recommended because the current evaluator dependency chain does not build cleanly on Python 3.14.

```powershell
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe catboost lightgbm xgboost
```

For full agent evaluation, copy `.env.example` to `.env`, add the API key for the configured model, and install Docker or Podman. Static validation and deterministic meta-evaluation do not require an API key.

## Validate and package

```powershell
.\.venv\Scripts\python.exe validate_submission.py --agent-dir submissions/01_robust_automl/agent
.\.venv\Scripts\python.exe scripts/package_submission.py
.\.venv\Scripts\python.exe scripts/package_submission.py --experiment 02_order_aware_automl
.\.venv\Scripts\python.exe scripts/package_submission.py --experiment 04_adaptive_automl_v3
.\.venv\Scripts\python.exe scripts/package_submission.py --experiment 05_selection_stable_automl_v4
```

The ZIP must contain `agent.yaml` at its root. Never zip the parent `agent/` directory itself.

## Meta-validation

Run a quick representative suite:

```powershell
.\.venv\Scripts\python.exe scripts/meta_evaluate.py train_13 train_06 train_16 train_11 --fast
```

Run all sixteen solved tasks:

```powershell
.\.venv\Scripts\python.exe scripts/meta_evaluate.py --fast
```

The evaluator reports full-test, inner-public, and inner-private AUC for every candidate. The supplied solutions are only used by the repository-level evaluation script; they are never bundled into the agent.

## Kaggle notebook

Upload `notebooks/build_agent_submission.ipynb` to Kaggle and run all cells. The notebook embeds the Agent Config source and writes:

```text
/kaggle/working/submission.zip
```

Regenerate the notebook after changing any agent file:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py
.\.venv\Scripts\python.exe scripts/build_notebook.py --experiment 02_order_aware_automl --output notebooks/build_agent_submission_v2.ipynb
.\.venv\Scripts\python.exe scripts/build_notebook.py --experiment 04_adaptive_automl_v3 --output notebooks/build_agent_submission_v3.ipynb
.\.venv\Scripts\python.exe scripts/build_notebook.py --experiment 05_selection_stable_automl_v4 --output notebooks/build_agent_submission_v4.ipynb
```

## Official local agent evaluation

After configuring `.env` and a container runtime:

```powershell
.\.venv\Scripts\python.exe run_local_eval.py `
  --submission-dir submissions/01_robust_automl/agent `
  --dataset train_01 `
  --metric roc_auc
```

## Competition submission

Accept the competition rules on Kaggle first, then upload `submissions/01_robust_automl/submission.zip` through the competition UI or Kaggle CLI.

## Design safeguards

- no symlinks or path traversal;
- no network access or runtime package installation inside the competition sandbox;
- no edits to the organizer's sample agent;
- deterministic random seeds;
- candidate cap below the 30-submission limit;
- public scores used only for coarse selection;
- exactly two final selections;
- fallback mode when optional boosting libraries fail.

## License

Code authored in this repository is released under the MIT License. The competition datasets and organizer starter materials retain their original CC BY 4.0 terms.
