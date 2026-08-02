# Autonomous Agent Prediction - Two-Stage Tabular Agent

This repository contains an Agent Config for Kaggle's Autonomous Agent Prediction beta competition. In each evaluation session, the agent solves two unseen binary-classification mini-competitions, submits prediction candidates through Kaggle's tools, and selects exactly two finalists.

## Current status

| Item | Status |
|---|---|
| Best official Kaggle score | **0.822 AUC** |
| Versions at the official best | v3-v7, v8.1, and v9-v12 |
| Current experiment | **v13 - public-led family refinement** |
| Agent source | [`submissions/15_public_refinement_v13/agent/`](submissions/15_public_refinement_v13/agent/) |
| Kaggle build notebook | [`notebooks/build_agent_submission_v13.ipynb`](notebooks/build_agent_submission_v13.ipynb) |
| Design notes | [`docs/V13.md`](docs/V13.md) |

The official score progressed from 0.781 to 0.814 and then 0.822. V4 through V7 remained at 0.822. V8 scored 0.500 after an execution-path failure, while the runtime-safe V8.1 restored 0.822. V9 through V12 also scored 0.822, despite testing full-data refits, a CV-gated MLP, equation discovery, and a count-view specialist.

V13 is deliberately different: it uses the public score as bounded feedback between two modeling stages. It is an experimental plateau-breaker, not a proven local upgrade. Its solved-task replay was approximately neutral against an older V8 meta snapshot, with useful gains on several generators and one large model-version mismatch on Train 13. Hidden Kaggle evaluation remains the deciding test.

## Score history

| Version | Official score | Main change |
|---|---:|---|
| v1 | 0.781 | Initial robust tabular AutoML agent |
| v2 | Invalid | Finished without calling `submit_predictions` |
| v2.1 | 0.814 | Guaranteed an initial valid submission and fixed persistent-work execution |
| v3 | **0.822** | Small-data CatBoost specialists, weighted blends, and explicit fallbacks |
| v4 | **0.822** | Two-finalist selection and routed numeric seed averaging |
| v5 | **0.822** | Regularized quadratic interactions |
| v6 | **0.822** | Routed XGBoost and Random Forest diversity |
| v7 | **0.822** | Dataset fingerprinting and deterministic baseline hedge |
| v8 | 0.500 | Runtime/execution failure left the chance-level fallback in place |
| v8.1 | **0.822** | Compact plan, bounded frontier, and immediate final selection |
| v9 | **0.822** | Conservative full-data refits and repeated CV |
| v10 | **0.822** | CV-gated neural specialist |
| v11 | **0.822** | Equation primitives gated against the historical hedge |
| v12 | **0.822** | OOF-gated count-view CatBoost lane |
| v13 | Pending | Public-led two-stage refinement with the CV hedge retained |

## V13 strategy

The LLM acts only as an orchestrator. Deterministic Python scripts perform the modeling and blending.

1. Submit `sample_submission.csv` immediately as a valid fallback.
2. Run a leakage-safe Stage 1 portfolio and emit p01-p13.
3. Reserve p01 for the strongest train-only CV hedge.
4. Submit the Stage 1 files and rank them by returned public AUC.
5. Pass the three public-leading filenames, plus p01, to `refine.py`.
6. Emit eight bounded rank blends around those leaders.
7. Allow a refined prediction to replace the Stage 1 public leader only after a public gain of at least 0.0001.
8. Select the public-led finalist and p01; if p01 is already the leader, use the highest-public other modeled submission.

The complete session uses at most 22 prediction submissions: one fallback, thirteen Stage 1 candidates, and eight refined candidates. This stays below Kaggle's limit of 30. Stage 2 only blends saved predictions, so a refinement failure cannot invalidate the Stage 1 result.

## Local evidence

The repository-level simulator reproduces the interaction loop using the Public and Private rows in each solved `solution.csv`.

- Stage 2 improved selected private AUC by +0.001981 on Train 09 and +0.001420 on Train 10.
- It also produced smaller positive results on Trains 03, 04, and 05.
- The full diagnostic replay was approximately neutral against the older V8 snapshot (`-0.00004` mean at the selected threshold).
- Train 13 exposed a model-version mismatch and candidate-frontier failure. The final agent appends shallow/ordered CatBoost and a specialist-excluding safety blend, improving that replay from 0.649242 to 0.650725, but it does not erase the mismatch with the older snapshot.
- The predeclared +0.002 mean-improvement target was not met. V13 should therefore be treated as a measured hidden-leaderboard experiment, not as a locally proven replacement for V12.

See [`docs/V13.md`](docs/V13.md) for exact policy details and artifact hashes.

## Validate and package V13

Python 3.13 is recommended because the evaluator dependency chain does not currently build cleanly on Python 3.14.

```powershell
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe catboost lightgbm xgboost
```

```powershell
.\.venv\Scripts\python.exe validate_submission.py `
  --agent-dir submissions/15_public_refinement_v13/agent

.\.venv\Scripts\python.exe scripts/package_submission.py `
  --experiment 15_public_refinement_v13
```

The local artifact is `submissions/15_public_refinement_v13/submission.zip`. The ZIP is intentionally ignored by Git and must contain `agent.yaml` at its root.

## Build in a Kaggle notebook

Upload [`notebooks/build_agent_submission_v13.ipynb`](notebooks/build_agent_submission_v13.ipynb) as a Kaggle notebook, run all cells, and download `/kaggle/working/submission.zip` from the notebook outputs.

Regenerate the notebook after any agent change:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py `
  --experiment 15_public_refinement_v13 `
  --output notebooks/build_agent_submission_v13.ipynb
```

## Reproduce the two-stage replay

Run selected tasks:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_v13_two_stage.py `
  train_03 train_09 train_13 `
  --output submissions/15_public_refinement_v13/replay.json
```

Replay replacement thresholds without retraining:

```powershell
python scripts/analyze_v13_policy.py `
  submissions/15_public_refinement_v13/pilot_train_01.json `
  submissions/15_public_refinement_v13/batch_a.json `
  submissions/15_public_refinement_v13/batch_b.json
```

The supplied `solution.csv` files are used only by repository-level evaluation scripts. They are never bundled into the Agent Config.

## Repository map

```text
submissions/01_robust_automl/                 v1
submissions/02_order_aware_automl/            v2
submissions/03_fail_safe_automl/              v2.1
submissions/04_adaptive_automl_v3/            v3
submissions/05_selection_stable_automl_v4/    v4
submissions/06_interaction_routed_automl_v5/  v5
submissions/07_tree_diversity_automl_v6/      v6
submissions/08_fingerprint_routed_automl_v7/  v7
submissions/09_meta_routed_automl_v8/         v8
submissions/10_runtime_safe_dgp_v81/          v8.1
submissions/11_full_refit_bagging_v9/         v9
submissions/12_cv_gated_mlp_v10/              v10
submissions/13_equation_discovery_v11/        v11
submissions/14_stacked_generalist_v12/        v12
submissions/15_public_refinement_v13/          v13 current experiment
notebooks/                                     self-contained Kaggle builders
scripts/                                       packaging and replay tools
docs/                                          version-specific design notes
kaggle-kaggle-skill/                           organizer documentation
```

## License

Code authored in this repository is released under the MIT License. Competition datasets and organizer starter materials retain their original CC BY 4.0 terms.
