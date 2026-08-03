# Autonomous Agent Prediction - Orthogonal Finalist Agent

This repository contains an Agent Config for Kaggle's Autonomous Agent Prediction beta competition. In each evaluation session, the agent solves two unseen binary-classification mini-competitions, submits prediction candidates, and selects exactly two finalists.

## Current status

| Item | Status |
|---|---|
| Best official Kaggle score | **0.822 AUC** |
| Versions at the official best | v3-v7 and v8.1-v13 |
| Current experiment | **v14 - orthogonal direct-model finalist** |
| Agent source | [`submissions/16_orthogonal_finalist_v14/agent/`](submissions/16_orthogonal_finalist_v14/agent/) |
| Kaggle build notebook | [`notebooks/build_agent_submission_v14.ipynb`](notebooks/build_agent_submission_v14.ipynb) |
| Design notes | [`docs/V14.md`](docs/V14.md) |

The official score progressed from 0.781 to 0.814 and then 0.822. V8 scored 0.500 after an execution failure, while V8.1 restored 0.822. Every valid experiment from V9 through V13 also scored exactly 0.822. V13's within-session public blend refinement did not break the plateau.

V14 changes how the second selection slot is used. It always retains the train-only p01 hedge, then forces the other finalist to be a direct model family with meaningful prediction disagreement. It does not spend the second slot on another ensemble close to p01.

## Score history

| Version | Official score | Main change |
|---|---:|---|
| v1 | 0.781 | Initial robust tabular AutoML agent |
| v2 | Invalid | Finished without calling `submit_predictions` |
| v2.1 | 0.814 | Guaranteed an initial valid submission and persistent-work execution |
| v3 | **0.822** | Small-data CatBoost specialists and explicit fallbacks |
| v4 | **0.822** | Two-finalist selection and numeric seed averaging |
| v5 | **0.822** | Regularized quadratic interactions |
| v6 | **0.822** | Routed XGBoost and Random Forest diversity |
| v7 | **0.822** | Dataset fingerprinting and deterministic baseline hedge |
| v8 | 0.500 | Execution failure left the chance-level fallback in place |
| v8.1 | **0.822** | Compact plan, bounded frontier, and immediate selection |
| v9 | **0.822** | Conservative full-data refits |
| v10 | **0.822** | CV-gated neural specialist |
| v11 | **0.822** | Equation primitives |
| v12 | **0.822** | OOF-gated count-view CatBoost lane |
| v13 | **0.822** | Public-led two-stage blend refinement |
| v14 | Pending | p01 plus a public-competitive orthogonal direct family |

## V14 strategy

1. Submit `sample_submission.csv` immediately as a valid fallback.
2. Run the established leakage-safe model portfolio and emit at most p01-p13.
3. Reserve p01 for the strongest train-only CV candidate.
4. Measure each direct model's rank disagreement with p01.
5. Submit every modeled candidate and identify the highest-public direct model.
6. Keep direct models within 0.005 public AUC of that direct-model leader.
7. Choose the eligible model with the greatest rank disagreement from p01.
8. Select p01 and that orthogonal direct model as the two finalists.

Direct families include CatBoost, LightGBM, ExtraTrees, Random Forest, histogram boosting, logistic/spline models, XGBoost, quadratic interactions, target encoding, and the small-data RBF kernel when routed. Ensembles are never eligible for the orthogonal slot.

The session uses at most 14 prediction submissions: one fallback and thirteen modeled candidates. V14 removes V13's Stage 2 script and eight public-tuned blend submissions.

## Local evidence

The policy is intentionally not optimized for mean solved-task AUC. Replaying the highest-public direct-family approximation reduced the V13 mean by about 0.00058, while p01 contained the loss on many tasks. This is the cost of turning the second slot into a high-disagreement bet.

Exact final-policy controls:

- Train 13 selected shallow CatBoost at diversity 0.0717 and scored 0.650786, slightly above V13's final 0.650725 replay.
- Train 09 selected logistic at diversity 0.2225, but p01 remained the better finalist at 0.651438, demonstrating the intended downside containment.
- On the saved Train 11 frontier, LightGBM is the only direct family within the public tolerance and improves selected private AUC from 0.825635 to 0.826487.
- The complete stdout plan was 395 characters on Train 13, below the 500-character capture limit.

V14 is a controlled hidden-leaderboard experiment. It assumes p01 was the stable contributor to the repeated 0.822 result; the official evaluation is the only way to confirm that assumption.

## Validate and package V14

Python 3.13 is recommended.

```powershell
.\.venv\Scripts\python.exe validate_submission.py `
  --agent-dir submissions/16_orthogonal_finalist_v14/agent

.\.venv\Scripts\python.exe scripts/package_submission.py `
  --experiment 16_orthogonal_finalist_v14
```

The generated artifact is `submissions/16_orthogonal_finalist_v14/submission.zip`. It must contain `agent.yaml` at the ZIP root.

## Build in a Kaggle notebook

Upload [`notebooks/build_agent_submission_v14.ipynb`](notebooks/build_agent_submission_v14.ipynb), run all cells, and download `/kaggle/working/submission.zip`.

Regenerate it after an agent change:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py `
  --experiment 16_orthogonal_finalist_v14 `
  --output notebooks/build_agent_submission_v14.ipynb
```

## Reproduce the selection tests

Replay the approximate policy without retraining:

```powershell
python scripts/analyze_v14_selection.py `
  submissions/15_public_refinement_v13/batch_a.json `
  submissions/15_public_refinement_v13/batch_b.json
```

Run the exact V14 policy on solved tasks:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_v14_orthogonal.py `
  train_09 train_13 `
  --output submissions/16_orthogonal_finalist_v14/targeted_results.json
```

The supplied `solution.csv` files are used only by repository-level evaluation scripts and are never bundled into the Agent Config.

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
submissions/15_public_refinement_v13/          v13
submissions/16_orthogonal_finalist_v14/        v14 current experiment
notebooks/                                     self-contained Kaggle builders
scripts/                                       packaging and replay tools
docs/                                          version-specific design notes
kaggle-kaggle-skill/                           organizer documentation
```

## License

Code authored in this repository is released under the MIT License. Competition datasets and organizer starter materials retain their original CC BY 4.0 terms.
