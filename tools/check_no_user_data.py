"""Fail if git would publish device-level data or notebook outputs.

Checks every tracked (or staged) file:

* data files are only allowed under ``demo/data/`` (synthetic) and
  ``results/`` (aggregates);
* tables under ``results/`` must not contain device identifiers and, where a
  device-count column exists, every row must reach ``min_devices_per_row``
  from ``configs/public_results.json``;
* notebooks must have no outputs (the nbstripout filter normally ensures this).

    python tools/check_no_user_data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_SUFFIXES = {".csv", ".parquet", ".geojson", ".jsonl", ".shp", ".gpkg", ".feather", ".pkl", ".zip", ".gz"}
ALLOWED_DATA_DIRS = ("demo/data/", "results/")
IDENTIFIER_COLUMNS = {"identifier", "device_id", "gaid", "idfa", "user_id"}
SYNTHETIC_PREFIX = "synthetic-"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def check_table(path: str, settings: dict) -> list[str]:
    full = ROOT / path
    if full.suffix == ".csv":
        table = pd.read_csv(full, nrows=200_000)
    elif full.suffix == ".parquet":
        table = pd.read_parquet(full)
    else:
        return []
    problems = []
    columns = {c.lower() for c in table.columns}
    found = columns & IDENTIFIER_COLUMNS
    if path.startswith("results/") and found:
        problems.append(f"{path}: device identifier column(s) {sorted(found)}")
    if path.startswith("demo/data/") and "identifier" in table.columns:
        real = ~table["identifier"].astype(str).str.startswith(SYNTHETIC_PREFIX)
        if real.any():
            problems.append(f"{path}: {int(real.sum())} identifiers are not synthetic")
    count_columns = {
        f"results/{dataset['target']}": dataset["device_count_column"] for dataset in settings["datasets"]
    }
    column = count_columns.get(path)
    threshold = settings["min_devices_per_row"]
    if column and (table[column] < threshold).any():
        problems.append(f"{path}: rows with {column} < {threshold}")
    return problems


def check_notebook(path: str) -> list[str]:
    notebook = json.loads((ROOT / path).read_text())
    with_outputs = sum(1 for cell in notebook.get("cells", []) if cell.get("outputs"))
    return [f"{path}: {with_outputs} cells with outputs"] if with_outputs else []


def main() -> int:
    settings = json.loads((ROOT / "configs/public_results.json").read_text())
    problems = []
    for path in tracked_files():
        suffix = Path(path).suffix.lower()
        if suffix == ".ipynb":
            problems += check_notebook(path)
        elif suffix in DATA_SUFFIXES:
            if not path.startswith(ALLOWED_DATA_DIRS):
                problems.append(f"{path}: data file outside {ALLOWED_DATA_DIRS}")
            else:
                problems += check_table(path, settings)
    print("\n".join(problems) or "No device-level data or notebook outputs tracked.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
