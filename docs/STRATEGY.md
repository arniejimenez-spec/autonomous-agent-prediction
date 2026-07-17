# Modeling and agent strategy

## Objective

Generalize across unseen datasets sampled from the same broad synthetic tabular family. Every training task is treated as a meta-validation fold: decisions should work across dataset sizes and schemas, not merely maximize one task.

## Observed task envelope

The sixteen supplied datasets range from 500 to 49,432 training rows, with 9–31 input columns, 0–25 categorical columns, and approximately 2–24% aggregate missingness. Every test set contains 10,000 rows and the classes are approximately balanced.

## Model portfolio

- CatBoost is the primary mixed-type learner and handles missing categorical structure naturally.
- LightGBM provides a different histogram-based inductive bias and is particularly effective on numeric-heavy tasks.
- ExtraTrees contributes high-variance nonlinear diversity.
- Regularized logistic regression captures additive and one-hot categorical signals that tree learners can underweight.
- Rank averaging makes model scales comparable while preserving AUC ordering.

## Validation

The skill uses deterministic shuffled stratified folds. Small and medium tasks use four folds; large or fast-mode tasks use three. Candidate ordering is based on out-of-fold AUC only. The public leaderboard is used to choose among precomputed candidates, not to manufacture more candidates.

## Agent budget

The main LLM uses a concise prompt and `gemini-3.1-flash-lite`, leaving most of the $2 allowance unused. One deterministic modeling call produces all candidates. The agent checks status before work and before final selection.

## Selection policy

The best public submission is paired with a meaningfully distinct high-CV candidate. This hedges inner-public noise and public leaderboard overfitting. When the diverse candidate is materially worse publicly, the two best public candidates are selected instead.

## Meta-validation results

Fast mode completed all sixteen solved tasks. The best generated candidate averaged **0.8021 full-test AUC**. Simulating selection by each task's 5,000-row inner-public split produced **0.8029 mean AUC** on the disjoint inner-private split. Per-task best AUC ranged from 0.6445 to 0.9685. Maximum modeling runtime was 274 seconds, well below the 60-minute session limit.

| Task | Best test AUC | Task | Best test AUC |
|---|---:|---|---:|
| train_01 | 0.7172 | train_09 | 0.6462 |
| train_02 | 0.9685 | train_10 | 0.8535 |
| train_03 | 0.8151 | train_11 | 0.8274 |
| train_04 | 0.8337 | train_12 | 0.7839 |
| train_05 | 0.6802 | train_13 | 0.6445 |
| train_06 | 0.8076 | train_14 | 0.8065 |
| train_07 | 0.8316 | train_15 | 0.8596 |
| train_08 | 0.8525 | train_16 | 0.9056 |

Raw candidate-level results are retained in `submissions/01_robust_automl/meta_results_all.json`.
