"""Create the root-correct submission archive on any platform."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="01_robust_automl")
    args = parser.parse_args()
    experiment = ROOT / "submissions" / args.experiment
    agent = experiment / "agent"
    destination = experiment / "submission.zip"
    if not (agent / "agent.yaml").is_file():
        raise FileNotFoundError(f"agent.yaml not found in {agent}")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(agent.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlink is not allowed: {path}")
            if path.is_file():
                archive.write(path, path.relative_to(agent).as_posix())
    print(destination)


if __name__ == "__main__":
    main()
