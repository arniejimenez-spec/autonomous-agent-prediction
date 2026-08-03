"""Evaluate V14's exact p01-plus-orthogonal selection on solved tasks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "submissions" / "16_orthogonal_finalist_v14"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DIRECT_FAMILIES = {
    "catboost", "catboost_shallow", "catboost_ordered", "lightgbm",
    "extra_trees", "random_forest", "histogram", "linear", "spline",
    "target_encoding", "quadratic", "xgboost", "rbf_kernel",
}


def score_file(path: Path, solution: pd.DataFrame) -> dict:
    prediction = pd.read_csv(path)
    identifiers = [
        column for column in prediction.columns
        if column in solution.columns and column not in {"target", "Usage"}
    ]
    merged = solution.merge(prediction, on=identifiers[0], suffixes=("_true", "_pred"))
    result = {"file": path.name}
    for usage, frame in merged.groupby("Usage"):
        result[usage.lower()] = roc_auc_score(frame["target_true"], frame["target_pred"])
    return result


def evaluate(dataset: str) -> dict:
    source = ROOT / "data" / dataset
    automl = EXPERIMENT / "agent" / "skills" / "tabular-automl" / "scripts" / "automl.py"
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_", dir=EXPERIMENT) as temporary:
        work = Path(temporary)
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copy2(source / name, work / name)
        completed = subprocess.run(
            [str(PYTHON), str(automl)], cwd=work, capture_output=True, text=True, timeout=3300,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)

        manifest = json.loads((work / "automl_manifest.json").read_text(encoding="utf-8"))
        solution = pd.read_csv(source / "solution.csv")
        scores = []
        for order, item in enumerate(manifest["candidates"]):
            scored = score_file(work / item["file"], solution)
            scored.update({
                "order": order,
                "name": item["name"],
                "family": item["family"],
                "cv_auc": item["cv_auc"],
                "diversity_from_hedge": item["diversity_from_hedge"],
                "printed_diversity": round(item["diversity_from_hedge"], 4),
            })
            scores.append(scored)

        hedge = next(item for item in scores if item["file"] == manifest["cv_hedge_file"])
        direct = [item for item in scores if item["family"] in DIRECT_FAMILIES]
        best_direct_public = max(item["public"] for item in direct)
        eligible = [item for item in direct if item["public"] >= best_direct_public - 0.005]
        orthogonal = max(
            eligible,
            key=lambda item: (
                item["printed_diversity"], item["public"], -item["order"],
            ),
        )
        selected_private = max(hedge["private"], orthogonal["private"])
        return {
            "dataset": dataset,
            "elapsed_seconds": manifest["elapsed_seconds"],
            "hedge": hedge,
            "orthogonal": orthogonal,
            "eligible": [item["file"] for item in eligible],
            "selected_private": selected_private,
            "scores": scores,
            "stdout": completed.stdout.strip(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", help="Solved datasets such as train_11")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "targeted_results.json")
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        result = evaluate(dataset)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            dataset, f"selected={result['selected_private']:.6f}",
            f"hedge={result['hedge']['file']}",
            f"orthogonal={result['orthogonal']['file']}",
            f"family={result['orthogonal']['family']}",
            f"diversity={result['orthogonal']['printed_diversity']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
