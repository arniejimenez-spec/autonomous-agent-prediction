"""Execute code cells from a generated build notebook in an isolated directory."""
from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import nbformat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook.resolve(), as_version=4)
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="agent-notebook-") as temporary:
        os.chdir(temporary)
        namespace = {"__name__": "__main__"}
        try:
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    exec(compile(cell.source, f"{args.notebook.name}:cell-{index}", "exec"), namespace)
            archive = Path("submission.zip")
            if not archive.is_file():
                raise FileNotFoundError("Notebook did not create submission.zip")
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                if "agent.yaml" not in names or any(name.startswith("agent/") for name in names):
                    raise ValueError("Notebook archive has an invalid root layout")
            print(f"Verified {len(names)} archive entries from {args.notebook}")
        finally:
            os.chdir(original)


if __name__ == "__main__":
    main()
