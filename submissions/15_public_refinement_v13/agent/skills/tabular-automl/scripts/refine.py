#!/usr/bin/env python3
"""Create a small public-led blend frontier from Stage 1 submissions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def enter_competition_workdir() -> None:
    configured = os.environ.get("KAGGLE_WORK_DIR")
    candidates = [Path.cwd()]
    if configured:
        candidates.append(Path(configured))
    candidates.extend([Path("/work"), Path("/kaggle/working")])
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("test.csv", "sample_submission.csv")):
            os.chdir(candidate)
            return
    raise FileNotFoundError("Competition work directory was not found")


def safe_file(value: str) -> Path:
    path = Path(value)
    if path.name != value or path.suffix.lower() != ".csv" or not path.is_file():
        raise ValueError(f"Invalid Stage 1 filename: {value}")
    return path


def rank01(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=float), method="average") / (len(values) + 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--third", required=True)
    parser.add_argument("--hedge", required=True)
    args = parser.parse_args()
    enter_competition_workdir()

    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")
    target_candidates = [column for column in sample.columns if column not in test.columns]
    target = target_candidates[-1] if target_candidates else sample.columns[-1]

    paths = {
        "first": safe_file(args.first),
        "second": safe_file(args.second),
        "third": safe_file(args.third),
        "hedge": safe_file(args.hedge),
    }
    predictions = {}
    for name, path in paths.items():
        frame = pd.read_csv(path)
        if len(frame) != len(sample) or target not in frame:
            raise ValueError(f"Malformed Stage 1 file: {path.name}")
        predictions[name] = rank01(frame[target].to_numpy())

    # These seven points search two leader/runner-up weights, a leader/third
    # frontier, and two conservative three-way mixtures. The public data only
    # chooses the inputs; labels never enter this script.
    recipes = [
        ("leader90_runner10", {"first": 0.90, "second": 0.10}),
        ("leader75_runner25", {"first": 0.75, "second": 0.25}),
        ("leader60_runner40", {"first": 0.60, "second": 0.40}),
        ("leader85_third15", {"first": 0.85, "third": 0.15}),
        ("leader65_third35", {"first": 0.65, "third": 0.35}),
        ("leader50_third50", {"first": 0.50, "third": 0.50}),
        ("leader65_runner25_third10", {"first": 0.65, "second": 0.25, "third": 0.10}),
        ("leader70_runner15_hedge15", {"first": 0.70, "second": 0.15, "hedge": 0.15}),
    ]

    outputs = []
    for index, (name, weights) in enumerate(recipes, start=1):
        blended = np.zeros(len(sample), dtype=float)
        for source, weight in weights.items():
            blended += weight * predictions[source]
        filename = f"r{index:02d}.csv"
        output = sample.copy()
        output[target] = np.clip(blended, 1e-7, 1 - 1e-7)
        output.to_csv(filename, index=False)
        outputs.append({"file": filename, "name": name, "weights": weights})

    manifest = {
        "inputs": {name: path.name for name, path in paths.items()},
        "candidates": outputs,
    }
    Path("refinement_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("REFINED " + " ".join(item["file"] for item in outputs))


if __name__ == "__main__":
    main()
