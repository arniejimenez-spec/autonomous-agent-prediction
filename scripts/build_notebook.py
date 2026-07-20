"""Build a self-contained Kaggle notebook embedding the current Agent Config."""
from __future__ import annotations

import base64
import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
def payload(agent: Path) -> dict[str, str]:
    result = {}
    for path in sorted(agent.rglob("*")):
        if path.is_file():
            result[path.relative_to(agent).as_posix()] = base64.b64encode(path.read_bytes()).decode("ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="01_robust_automl")
    parser.add_argument("--output")
    args = parser.parse_args()
    agent = ROOT / "submissions" / args.experiment / "agent"
    output = Path(args.output) if args.output else ROOT / "notebooks" / "build_agent_submission.ipynb"
    files = json.dumps(payload(agent), sort_keys=True)
    nb = nbf.v4.new_notebook()
    nb["metadata"].update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "none", "internet": False, "isGpuEnabled": False},
    })
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Build Autonomous Agent Prediction submission\n\n"
            "This self-contained notebook reconstructs the validated Agent Config and creates "
            "`/kaggle/working/submission.zip`. No internet or dataset attachment is required."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport base64, json, shutil, zipfile\n\n"
            f"FILES = json.loads({json.dumps(files)})\n"
            "work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()\n"
            "agent_dir = work / 'agent'\n"
            "if agent_dir.exists():\n    shutil.rmtree(agent_dir)\n"
            "agent_dir.mkdir(parents=True)\n"
            "for relative, encoded in FILES.items():\n"
            "    destination = agent_dir / relative\n"
            "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "    destination.write_bytes(base64.b64decode(encoded))\n"
            "print(f'Restored {len(FILES)} files to {agent_dir}')"
        ),
        nbf.v4.new_code_cell(
            "zip_path = work / 'submission.zip'\n"
            "if zip_path.exists():\n    zip_path.unlink()\n"
            "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:\n"
            "    for path in sorted(agent_dir.rglob('*')):\n"
            "        if path.is_file():\n"
            "            archive.write(path, path.relative_to(agent_dir).as_posix())\n"
            "with zipfile.ZipFile(zip_path) as archive:\n"
            "    names = archive.namelist()\n"
            "assert 'agent.yaml' in names and all(not n.startswith('agent/') for n in names)\n"
            "print(f'Created {zip_path} ({zip_path.stat().st_size:,} bytes)')\n"
            "print('\\n'.join(names))"
        ),
        nbf.v4.new_markdown_cell(
            "The notebook output named `submission.zip` is the artifact to submit to the competition."
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(output)


if __name__ == "__main__":
    main()
