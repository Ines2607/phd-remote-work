"""Fail if library or pipeline code contains values that belong in configs.

Looks for absolute or cloud paths, BigQuery projects / tables, EPSG codes,
literal YYYYMM months and region names in ``remote_work_utils/`` and
``pipelines/``. Legacy reference copies under ``research/`` are not checked.

    python tools/check_hardcoded.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = ["remote_work_utils", "pipelines", "tools"]
SKIP = {"tools/check_hardcoded.py"}

PATTERNS = {
    "absolute path": re.compile(r"""["'](/Users/|/home/|C:\\\\)"""),
    "cloud path": re.compile(r"""["']gs://"""),
    "BigQuery project or table": re.compile(r"phd-habidatum|phd_[a-z_]+\.[a-z_]+"),
    "EPSG code": re.compile(r"""(EPSG:\d{4,5}|epsg=\d{4,5}|crs\s*=\s*\d{4,5})""", re.IGNORECASE),
    "literal month": re.compile(r"""["'](19|20)\d{2}(0[1-9]|1[0-2])["']"""),
    "region name": re.compile(r"Gush ?Dan|Tel ?Aviv|yishuv_stat|SHEM_YISHUV", re.IGNORECASE),
}
# Geographic WGS84 (EPSG:4326) is the format of the inputs, not a region choice.
ALLOWED = [re.compile(r"crs\s*=\s*4326|EPSG:4326", re.IGNORECASE)]


def scan(path: Path) -> list[str]:
    problems = []
    in_docstring = False
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.count('"""') == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith("#") or stripped.startswith('"""'):
            continue
        cleaned = line
        for allowed in ALLOWED:
            cleaned = allowed.sub("", cleaned)
        for name, pattern in PATTERNS.items():
            if pattern.search(cleaned):
                problems.append(f"{path.relative_to(ROOT)}:{number}: {name}: {stripped}")
    return problems


def main() -> int:
    problems = []
    for folder in CHECKED:
        for path in sorted((ROOT / folder).rglob("*.py")):
            if str(path.relative_to(ROOT)) not in SKIP:
                problems += scan(path)
    print("\n".join(problems) or "No hardcoded values found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
