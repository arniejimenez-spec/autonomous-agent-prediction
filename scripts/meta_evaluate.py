"""Run the bundled AutoML script directly against solved training tasks."""
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
def evaluate(dataset: str, fast: bool, experiment: str) -> dict:
    source = ROOT / "data" / dataset
    experiment_dir = ROOT / "submissions" / experiment
    automl = experiment_dir / "agent/skills/tabular-automl/scripts/automl.py"
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_", dir=experiment_dir) as tmp:
        work = Path(tmp)
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copy2(source / name, work / name)
        cmd = [str(ROOT / ".venv/Scripts/python.exe"), str(automl)]
        if fast:
            cmd.append("--fast")
        completed = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3300)
        if completed.returncode:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)
        manifest = json.loads((work / "automl_manifest.json").read_text())
        solution = pd.read_csv(source / "solution.csv")
        scores = []
        for item in manifest["candidates"]:
            pred = pd.read_csv(work / item["file"])
            merged = solution.merge(pred, on="row_id", suffixes=("_true", "_pred"))
            overall = roc_auc_score(merged["target_true"], merged["target_pred"])
            split = {usage: roc_auc_score(g["target_true"], g["target_pred"])
                     for usage, g in merged.groupby("Usage")}
            scores.append({"file": item["file"], "name": item.get("name"),
                           "members": item.get("members"), "cv_auc": item["cv_auc"],
                           "test_auc": overall, **{f"test_{k.lower()}": v for k, v in split.items()}})
        return {
            "dataset": dataset,
            "elapsed_seconds": manifest["elapsed_seconds"],
            "scores": scores,
            "v9_variants": manifest.get("v9_variants", []),
            "v10_specialist": manifest.get("v10_specialist"),
            "v11_specialist": manifest.get("v11_specialist"),
            "cv_hedge_file": manifest.get("cv_hedge_file"),
            "stdout": completed.stdout,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=[f"train_{i:02d}" for i in range(1, 17)])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--experiment", default="01_robust_automl")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output or f"submissions/{args.experiment}/meta_results.json")
    results = []
    for dataset in args.datasets:
        result = evaluate(dataset, args.fast, args.experiment)
        results.append(result)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        best = max(result["scores"], key=lambda x: x["test_auc"])
        print(dataset, f"best_test_auc={best['test_auc']:.6f}", best["file"], flush=True)


if __name__ == "__main__":
    main()
