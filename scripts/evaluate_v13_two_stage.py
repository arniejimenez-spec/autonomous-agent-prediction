"""Simulate V13's public-feedback loop on solved meta-training datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "submissions" / "15_public_refinement_v13"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def score_file(path: Path, solution: pd.DataFrame) -> dict:
    prediction = pd.read_csv(path)
    id_columns = [column for column in prediction.columns if column in solution.columns and column not in {"target", "Usage"}]
    identifier = id_columns[0]
    merged = solution.merge(prediction, on=identifier, suffixes=("_true", "_pred"))
    result = {"file": path.name}
    for usage, frame in merged.groupby("Usage"):
        result[usage.lower()] = roc_auc_score(frame["target_true"], frame["target_pred"])
    return result


def baseline_selected(record: dict) -> float:
    public = sorted(record["scores"], key=lambda item: item["test_public"], reverse=True)
    cv = sorted(record["scores"], key=lambda item: item["cv_auc"], reverse=True)
    primary = public[0]
    hedge = cv[0] if cv[0]["file"] != primary["file"] else public[1]
    return max(primary["test_private"], hedge["test_private"])


def evaluate(dataset: str, threshold: float) -> dict:
    source = ROOT / "data" / dataset
    automl = EXPERIMENT / "agent" / "skills" / "tabular-automl" / "scripts" / "automl.py"
    refine = EXPERIMENT / "agent" / "skills" / "tabular-automl" / "scripts" / "refine.py"
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_", dir=EXPERIMENT) as temporary:
        work = Path(temporary)
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copy2(source / name, work / name)
        first_run = subprocess.run(
            [str(PYTHON), str(automl)], cwd=work, capture_output=True, text=True, timeout=3300,
        )
        if first_run.returncode:
            raise RuntimeError(first_run.stdout + "\n" + first_run.stderr)

        manifest = json.loads((work / "automl_manifest.json").read_text(encoding="utf-8"))
        solution = pd.read_csv(source / "solution.csv")
        stage1 = []
        metadata = {item["file"]: item for item in manifest["candidates"]}
        for item in manifest["candidates"]:
            scored = score_file(work / item["file"], solution)
            scored.update({
                "name": item["name"], "family": item["family"], "cv_auc": item["cv_auc"],
            })
            stage1.append(scored)
        ranked = sorted(enumerate(stage1), key=lambda pair: (-pair[1]["public"], pair[0]))
        top = [item["file"] for _, item in ranked[:3]]
        while len(top) < 3:
            top.append(top[-1])
        hedge_file = manifest["cv_hedge_file"]

        second_run = subprocess.run(
            [
                str(PYTHON), str(refine),
                "--first", top[0], "--second", top[1], "--third", top[2],
                "--hedge", hedge_file,
            ],
            cwd=work, capture_output=True, text=True, timeout=300,
        )
        if second_run.returncode:
            raise RuntimeError(second_run.stdout + "\n" + second_run.stderr)
        refine_manifest = json.loads((work / "refinement_manifest.json").read_text(encoding="utf-8"))
        refined = []
        refine_metadata = {item["file"]: item for item in refine_manifest["candidates"]}
        for item in refine_manifest["candidates"]:
            scored = score_file(work / item["file"], solution)
            scored.update({"name": item["name"], "weights": item["weights"]})
            refined.append(scored)

        stage1_leader = max(enumerate(stage1), key=lambda pair: (pair[1]["public"], -pair[0]))[1]
        refined_leader = max(enumerate(refined), key=lambda pair: (pair[1]["public"], -pair[0]))[1]
        public_finalist = (
            refined_leader
            if refined_leader["public"] >= stage1_leader["public"] + threshold
            else stage1_leader
        )
        hedge = next(item for item in stage1 if item["file"] == hedge_file)
        if public_finalist["file"] == hedge["file"]:
            hedge = next(item for _, item in ranked if item["file"] != public_finalist["file"])
        selected_private = max(public_finalist["private"], hedge["private"])
        oracle_private = max(item["private"] for item in stage1 + refined)
        return {
            "dataset": dataset,
            "elapsed_seconds": manifest["elapsed_seconds"],
            "stage1_inputs": top,
            "stage1": stage1,
            "refined": refined,
            "stage1_leader": stage1_leader["file"],
            "refined_leader": refined_leader["file"],
            "public_finalist": public_finalist["file"],
            "hedge": hedge["file"],
            "selected_private": selected_private,
            "oracle_private": oracle_private,
            "stdout": {"stage1": first_run.stdout.strip(), "refine": second_run.stdout.strip()},
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=[f"train_{index:02d}" for index in range(1, 17)])
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--baseline", type=Path, default=ROOT / "submissions" / "09_meta_routed_automl_v8" / "meta_results_all.json")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "meta_results_all.json")
    args = parser.parse_args()
    baseline = {
        record["dataset"]: baseline_selected(record)
        for record in json.loads(args.baseline.read_text(encoding="utf-8"))
    }
    results = []
    for dataset in args.datasets:
        result = evaluate(dataset, args.threshold)
        result["baseline_selected_private"] = baseline.get(dataset)
        result["baseline_delta"] = (
            result["selected_private"] - baseline[dataset] if dataset in baseline else None
        )
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            dataset,
            f"selected={result['selected_private']:.6f}",
            f"delta={result['baseline_delta']:+.6f}" if result["baseline_delta"] is not None else "",
            f"public={result['public_finalist']}", f"hedge={result['hedge']}",
            flush=True,
        )
    deltas = [result["baseline_delta"] for result in results if result["baseline_delta"] is not None]
    print("mean_selected_private", f"{statistics.mean(result['selected_private'] for result in results):.6f}")
    if deltas:
        print("mean_baseline_delta", f"{statistics.mean(deltas):+.6f}")
        print("regressions", sum(delta < -1e-12 for delta in deltas))


if __name__ == "__main__":
    main()
