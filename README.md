# Autonomous Agent Prediction — Robust Tabular AutoML Agent

This repository contains a competition-ready Agent Config for Kaggle's Autonomous Agent Prediction beta competition. The agent solves an unseen binary tabular task inside a 60-minute CPU sandbox, uses public leaderboard feedback conservatively, and selects two final submissions.

## What is included

- `submissions/01_robust_automl/agent/` — uploadable Agent Config source
- `submissions/01_robust_automl/submission.zip` — locally generated Kaggle artifact (excluded from Git; rebuilt by the notebook or CI)
- `notebooks/build_agent_submission.ipynb` — self-contained Kaggle notebook that rebuilds the ZIP
- `scripts/meta_evaluate.py` — evaluates the deterministic ML skill on the 16 solved tasks
- `scripts/build_notebook.py` — regenerates the self-contained notebook from the agent directory
- `scripts/package_submission.py` — cross-platform Agent Config packager
- `scripts/package_submission.ps1` — packages the Agent Config on Windows
- `kaggle-kaggle-skill/` — organizer-supplied competition documentation

The organizer-supplied `sample_submission/` is intentionally unchanged.

## Architecture

The LLM is a low-cost orchestration layer. Expensive and error-prone modeling work lives in a pre-tested sandbox skill:

1. infer the target, ID, feature types, and binary label mapping;
2. run stratified cross-validation across CatBoost, LightGBM, ExtraTrees, and regularized logistic regression;
3. generate leakage-safe out-of-fold predictions;
4. create greedy and broad rank blends;
5. write ranked candidate CSV files and a machine-readable manifest;
6. submit at most six candidates and select two robust, distinct finalists.

This structure spends LLM tokens on orchestration instead of generating ad hoc training code. `gemini-3.1-flash-lite` was chosen because the official $2 session budget rewards a concise, deterministic workflow.

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
