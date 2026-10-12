#!/usr/bin/env python3
# flake8: noqa
"""
Generate lightweight Markdown documentation for every .ipynb in the repo.

Outputs:
  - notebook_docs/<notebook-path>.md  (mirrors directory structure)
  - notebook_docs/INDEX.md            (run order + dependency links)

Docs format per notebook (as requested):
  - Goal
  - Main steps
  - Outputs

Additionally we include:
  - Inputs (helps order notebooks)
  - Runs before / Runs after (best-effort, inferred)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


WRITE_PATTERNS = [
    # pandas / geopandas / matplotlib / plotly
    r"\.to_csv\(",
    r"\.to_parquet\(",
    r"\.to_pickle\(",
    r"\.to_excel\(",
    r"\.to_feather\(",
    r"\.to_json\(",
    r"\.to_file\(",
    r"\.savefig\(",
    r"\.write_image\(",
    r"\.write_html\(",
    r"joblib\.dump\(",
    r"pickle\.dump\(",
    r"open\([^,]+,\s*['\"]w",
    # bigquery / gbq
    r"\.to_gbq\(",
    r"CREATE\s+TABLE",
    r"CREATE\s+OR\s+REPLACE\s+TABLE",
]

READ_PATTERNS = [
    r"pd\.read_csv\(",
    r"pd\.read_parquet\(",
    r"pd\.read_pickle\(",
    r"pd\.read_excel\(",
    r"pd\.read_feather\(",
    r"pd\.read_json\(",
    r"gpd\.read_file\(",
    r"open\([^,]+,\s*['\"]r",
    r"read_gbq\(",
    r"FROM\s+`",  # bigquery SQL
]


STRING_LITERAL_RE = re.compile(
    r"""
    (?P<q>["'])
    (?P<s>(?:\\.|(?!\1).)*)
    (?P=q)
    """,
    re.VERBOSE,
)

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")

GENERIC_HEADINGS = {
    "description",
    "description:",
    "goal",
    "goal:",
    "overview",
    "overview:",
    "todo",
    "todo:",
}


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _normalize_path_str(p: str) -> str:
    p = p.strip()
    # drop f-string braces and format placeholders; keep as informational path token
    p = re.sub(r"\{[^}]+\}", "{...}", p)
    p = p.replace("\\", "/")
    return p


def _extract_string_literals(code: str) -> list[str]:
    out: list[str] = []
    for m in STRING_LITERAL_RE.finditer(code):
        s = m.group("s")
        if not s:
            continue
        s = s.replace("\\n", "\n")
        out.append(s)
    return out


def _likely_path_token(s: str) -> bool:
    s = s.strip()
    if len(s) < 3:
        return False
    # Drop nonsense like ".csv" / ".parquet"
    if re.fullmatch(r"\.[a-z0-9]{2,8}", s, re.I):
        return False
    # very lightweight heuristics: has a slash or a dot-extension
    if "/" in s:
        return True
    if re.search(r"\.(csv|parquet|pkl|pickle|xlsx|xls|geojson|json|html|png|jpg|jpeg|pdf|gpkg|shp|feather)\b", s, re.I):
        # Require some basename-ish characters before the extension (avoid ".csv")
        if re.fullmatch(r"\.[a-z0-9]{2,8}", s, re.I):
            return False
        return True
    return False


def _canonicalize_path_literal(nb_path: Path, literal: str) -> str:
    """
    Convert a string literal that looks like a path into a stable representation.
    If it's a relative path, resolve relative to the notebook directory.
    If it contains env vars or placeholders, keep as-is but normalized.
    """
    lit = _normalize_path_str(literal)
    if re.search(r"(\$|\{\.{3}\}|\{|\}|%s|%d)", lit):
        return lit
    if re.match(r"^[a-zA-Z]+://", lit):
        return lit
    if lit.startswith("/"):
        return lit
    # relative path: resolve against notebook parent
    try:
        resolved = (nb_path.parent / lit).resolve()
        return str(resolved)
    except Exception:
        return lit


def _extract_headings_from_markdown(md: str) -> list[str]:
    headings: list[str] = []
    for line in md.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        text = m.group(2).strip()
        # skip extremely generic titles / container headings
        if not text or text.lower() in {"title"} or text.lower() in GENERIC_HEADINGS:
            continue
        if level <= 3:
            headings.append(text)
    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for h in headings:
        k = h.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


def _first_nonempty_markdown(cells: list[dict]) -> Optional[str]:
    for c in cells:
        if c.get("cell_type") != "markdown":
            continue
        src = "".join(c.get("source") or "").strip()
        if src:
            return src
    return None


def _infer_goal(nb_path: Path, cells: list[dict]) -> str:
    md0 = _first_nonempty_markdown(cells) or ""
    # If there is a top-level heading, use the first one as goal-ish.
    raw_headings = []
    for line in md0.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        raw_headings.append(m.group(2).strip())

    # Special case: notebooks that start with "## Description:" or similar.
    if raw_headings:
        first = raw_headings[0].strip()
        if first.lower() not in GENERIC_HEADINGS:
            return first

    headings = _extract_headings_from_markdown(md0)
    if headings:
        return headings[0]
    # Otherwise use first paragraph-ish line
    for line in md0.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower() in GENERIC_HEADINGS:
            continue
        # avoid headings markers
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        if line.lower() in GENERIC_HEADINGS:
            continue
        if line:
            return line[:240]
    # Fall back to filename (humanized)
    name = nb_path.stem.replace("_", " ").replace("-", " ").strip()
    return name


def _infer_main_steps(cells: list[dict]) -> list[str]:
    steps: list[str] = []
    for c in cells:
        if c.get("cell_type") != "markdown":
            continue
        src = "".join(c.get("source") or "")
        steps.extend(_extract_headings_from_markdown(src))
    # If no headings, summarize with coarse-grained code actions
    if steps:
        return steps[:20]

    # Fallback: classify common action verbs from code cells
    actions: list[str] = []
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        code = _strip_ansi("".join(c.get("source") or ""))
        if re.search(r"kmeans|sklearn|RandomForest|XGBoost|lightgbm|statsmodels", code, re.I):
            actions.append("Fit / evaluate statistical or ML models")
        if re.search(r"(groupby|agg|merge|join|pivot|resample)", code):
            actions.append("Transform and aggregate data (merge/groupby/pivot)")
        if re.search(r"(plot|plt\.|seaborn|sns\.|altair|plotly)", code, re.I):
            actions.append("Visualize results (plots/maps)")
        if re.search(r"(to_csv|to_parquet|to_pickle|to_excel|to_file|savefig|write_html|write_image)", code):
            actions.append("Export outputs to disk (tables/figures)")

    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for a in actions:
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out[:10] if out else ["(No clear section headings found; see notebook for details)"]


def _extract_paths_from_outputs(nb_path: Path, outputs: list[dict]) -> list[str]:
    """
    Pull path-like tokens from already-executed cell outputs (stdout, errors, rich repr).
    This often contains the clearest "saved to ..." messages even when source code
    writes via helper functions / scripts.
    """
    found: list[str] = []
    for out in outputs or []:
        # stdout/stderr streams
        if isinstance(out, dict):
            txt_parts: list[str] = []
            if "text" in out and isinstance(out["text"], str):
                txt_parts.append(out["text"])
            if "text" in out and isinstance(out["text"], list):
                txt_parts.extend([t for t in out["text"] if isinstance(t, str)])
            if "ename" in out and "evalue" in out:
                txt_parts.append(str(out.get("evalue", "")))
            if "data" in out and isinstance(out["data"], dict):
                for v in out["data"].values():
                    if isinstance(v, str):
                        txt_parts.append(v)
                    elif isinstance(v, list):
                        txt_parts.extend([t for t in v if isinstance(t, str)])
            text = "\n".join(txt_parts)
        else:
            text = str(out)

        # Find explicit file paths or relative paths with common extensions
        for m in re.finditer(
            r"((?:/|\.{0,2}/)[^\s'\"`]+?\.(?:csv|parquet|pkl|pickle|xlsx|xls|geojson|json|html|png|jpg|jpeg|pdf|gpkg|shp|feather))",
            text,
            re.I,
        ):
            lit = m.group(1)
            if _likely_path_token(lit):
                found.append(_canonicalize_path_literal(nb_path, lit))

    # de-dup preserving order
    seen = set()
    out: list[str] = []
    for x in found:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _extract_io(cells: list[dict], nb_path: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Return (inputs, outputs, bq_tables) as best-effort lists.
    Inputs/outputs are canonicalized absolute paths when resolvable.
    """
    inputs: list[str] = []
    outputs: list[str] = []
    bq_tables: list[str] = []

    write_re = re.compile("|".join(WRITE_PATTERNS), re.I)
    read_re = re.compile("|".join(READ_PATTERNS), re.I)

    for c in cells:
        if c.get("cell_type") != "code":
            continue
        code = _strip_ansi("".join(c.get("source") or ""))

        # BigQuery tables referenced in SQL backticks: FROM `project.dataset.table`
        for m in re.finditer(r"`([^`]+\.[^`]+)`", code):
            token = m.group(1).strip()
            if token.count(".") >= 1:
                bq_tables.append(token)

        if read_re.search(code):
            for s in _extract_string_literals(code):
                if _likely_path_token(s):
                    inputs.append(_canonicalize_path_literal(nb_path, s))

        if write_re.search(code):
            for s in _extract_string_literals(code):
                if _likely_path_token(s):
                    outputs.append(_canonicalize_path_literal(nb_path, s))

        # Also mine already-executed outputs for path-like tokens (often contains "saved ...")
        try:
            outputs.extend(_extract_paths_from_outputs(nb_path, c.get("outputs") or []))
        except Exception:
            pass

    def uniq(xs: Iterable[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for x in xs:
            x = x.strip()
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return uniq(inputs), uniq(outputs), uniq(bq_tables)


def _classify_notebook(nb_rel: str) -> str:
    rel_lower = nb_rel.lower()
    name_lower = Path(nb_rel).name.lower()
    if "demo" in rel_lower or "demo" in name_lower:
        return "demo"
    if "/research-" in rel_lower or rel_lower.startswith("100-research") or rel_lower.startswith("200-research") or rel_lower.startswith("300-research") or rel_lower.startswith("400-research") or rel_lower.startswith("500-research") or rel_lower.startswith("700-research") or rel_lower.startswith("800-research"):
        return "research"
    if "/process-" in rel_lower or rel_lower.startswith("010-process") or rel_lower.startswith("020-process") or rel_lower.startswith("030-process") or rel_lower.startswith("060-process") or rel_lower.startswith("300-process"):
        return "process"
    if "/load-" in rel_lower or rel_lower.startswith("001-load") or rel_lower.startswith("002-load") or rel_lower.startswith("003-load") or rel_lower.startswith("004-load") or rel_lower.startswith("005-load") or rel_lower.startswith("006-load") or rel_lower.startswith("007-load"):
        return "load"
    if rel_lower.startswith("900-features"):
        return "features"
    return "other"


def _sort_key_for_run_order(relpath: str) -> tuple:
    # prefer numeric prefixes in directories and filenames
    parts = Path(relpath).parts
    key: list[tuple[int, str]] = []
    for p in parts:
        m = re.match(r"^(\d{1,4})[-_].*", p)
        if m:
            key.append((int(m.group(1)), p))
        else:
            key.append((10_000, p))
    return tuple(key)


@dataclass
class NotebookInfo:
    path: Path
    rel: str
    nb_type: str
    goal: str
    steps: list[str]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    bq_tables: list[str] = field(default_factory=list)


def _load_notebook(nb_path: Path) -> dict:
    with nb_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_notebook_md(
    root: Path,
    docs_root: Path,
    info: NotebookInfo,
    runs_after: list[str],
    runs_before: list[str],
    notebook_link_from_doc: str,
) -> str:
    rel_display = _safe_relpath(info.path, root)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    def relize_paths(paths: list[str]) -> list[str]:
        out: list[str] = []
        for p in paths:
            # convert absolute resolved paths back to repo-relative when possible
            try:
                pp = Path(p)
                if pp.is_absolute():
                    out.append(_safe_relpath(pp, root))
                else:
                    out.append(p)
            except Exception:
                out.append(p)
        return out

    inputs_rel = relize_paths(info.inputs)
    outputs_rel = relize_paths(info.outputs)

    lines: list[str] = []
    lines.append(f"# Notebook documentation: `{rel_display}`")
    lines.append("")
    lines.append(f"- **Notebook**: `{rel_display}`")
    lines.append(f"- **Doc generated**: {generated_at}")
    lines.append(f"- **Type**: `{info.nb_type}`")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(info.goal.strip() or "(Not specified)")
    lines.append("")
    lines.append("## Main steps")
    lines.append("")
    for s in info.steps:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    if outputs_rel or info.bq_tables:
        if outputs_rel:
            lines.append("### Files")
            lines.append("")
            for p in outputs_rel[:50]:
                lines.append(f"- `{p}`")
            if len(outputs_rel) > 50:
                lines.append(f"- (and {len(outputs_rel) - 50} more…)")  # avoid huge pages
            lines.append("")
        if info.bq_tables:
            lines.append("### BigQuery tables (referenced)")
            lines.append("")
            for t in info.bq_tables[:50]:
                lines.append(f"- `{t}`")
            if len(info.bq_tables) > 50:
                lines.append(f"- (and {len(info.bq_tables) - 50} more…)")  # avoid huge pages
            lines.append("")
    else:
        lines.append("(No obvious outputs detected from static analysis; notebook may primarily plot/display results or write via helper functions.)")
        lines.append("")

    # Helpful for ordering, but keep it small.
    lines.append("## Inputs")
    lines.append("")
    if inputs_rel:
        for p in inputs_rel[:50]:
            lines.append(f"- `{p}`")
        if len(inputs_rel) > 50:
            lines.append(f"- (and {len(inputs_rel) - 50} more…)")
    else:
        lines.append("(No obvious file inputs detected from static analysis.)")
    lines.append("")

    lines.append("## Suggested ordering (best-effort)")
    lines.append("")
    if runs_after:
        lines.append("### Runs after")
        lines.append("")
        for r in runs_after:
            lines.append(f"- {r}")
        lines.append("")
    if runs_before:
        lines.append("### Runs before")
        lines.append("")
        for r in runs_before:
            lines.append(f"- {r}")
        lines.append("")
    if not runs_after and not runs_before:
        lines.append("(No clear predecessor/successor notebooks inferred.)")
        lines.append("")

    lines.append("## Links")
    lines.append("")
    lines.append(f"- Open notebook: `{notebook_link_from_doc}`")
    lines.append("")
    return "\n".join(lines)


def _render_index_md(root: Path, docs_root: Path, infos: list[NotebookInfo], edges_after: dict[str, list[str]], edges_before: dict[str, list[str]]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# Notebook documentation index")
    lines.append("")
    lines.append(f"- **Generated**: {generated_at}")
    lines.append(f"- **Repo root**: `{root}`")
    lines.append(f"- **Notebooks found**: {len(infos)}")
    lines.append("")
    lines.append("## Suggested run order")
    lines.append("")
    lines.append("This ordering is inferred from directory/file numeric prefixes plus detected file I/O dependencies. Treat it as a starting point; some notebooks are exploratory (`research`) or public examples (`demo`).")
    lines.append("")

    # Group by top-level folder
    by_folder: dict[str, list[NotebookInfo]] = {}
    for info in infos:
        top = Path(info.rel).parts[0] if Path(info.rel).parts else info.rel
        by_folder.setdefault(top, []).append(info)

    for top in sorted(by_folder.keys(), key=_sort_key_for_run_order):
        group = sorted(by_folder[top], key=lambda x: _sort_key_for_run_order(x.rel))
        lines.append(f"### `{top}`")
        lines.append("")
        for info in group:
            doc_rel_from_index = f"./{Path(info.rel).with_suffix('.md')}"
            nb_rel_from_index = f"../{info.rel}"
            nb_rel = info.rel
            nb_type = info.nb_type
            ra = edges_after.get(nb_rel, [])
            rb = edges_before.get(nb_rel, [])
            suffix_bits: list[str] = [f"`{nb_type}`"]
            if ra:
                suffix_bits.append(f"after: {len(ra)}")
            if rb:
                suffix_bits.append(f"before: {len(rb)}")
            suffix = ", ".join(suffix_bits)
            lines.append(f"- [`{nb_rel}`]({nb_rel_from_index}) → doc: [{doc_rel_from_index}]({doc_rel_from_index}) ({suffix})")
        lines.append("")

    # Also add dependency spotlight: notebooks with many edges
    lines.append("## Dependency highlights")
    lines.append("")
    scored = []
    for info in infos:
        nb = info.rel
        scored.append((len(edges_after.get(nb, [])) + len(edges_before.get(nb, [])), nb))
    scored.sort(reverse=True)
    for score, nb in scored[:25]:
        if score == 0:
            break
        lines.append(f"- **`{nb}`**: {score} inferred links (after: {len(edges_after.get(nb, []))}, before: {len(edges_before.get(nb, []))})")
    lines.append("")

    return "\n".join(lines)


def _render_all_notebooks_md(
    root: Path,
    docs_root: Path,
    infos: list[NotebookInfo],
    edges_after: dict[str, list[str]],
    edges_before: dict[str, list[str]],
) -> str:
    """
    Single, consolidated markdown for all notebooks, as requested.
    Kept reasonably compact by truncating long IO lists.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# All notebook documentation")
    lines.append("")
    lines.append(f"- **Generated**: {generated_at}")
    lines.append(f"- **Notebooks found**: {len(infos)}")
    lines.append("")
    lines.append("## Suggested run order (high-level)")
    lines.append("")
    lines.append("Best-effort ordering based on numeric prefixes + detected I/O dependencies. Use `notebook_docs/INDEX.md` for a folder-grouped view.")
    lines.append("")
    lines.append("- Index: [INDEX.md](./INDEX.md)")
    lines.append("")
    lines.append("## Table of contents")
    lines.append("")
    for info in infos:
        # Use links to headings within this doc.
        # GitHub-style anchor is lowercased with non-alnum -> '-'
        anchor = re.sub(r"[^a-z0-9]+", "-", info.rel.lower()).strip("-")
        lines.append(f"- [`{info.rel}`](#{anchor})")
    lines.append("")

    def relize_paths(paths: list[str]) -> list[str]:
        out: list[str] = []
        for p in paths:
            try:
                pp = Path(p)
                if pp.is_absolute():
                    out.append(_safe_relpath(pp, root))
                else:
                    out.append(p)
            except Exception:
                out.append(p)
        return out

    for info in infos:
        rel_display = info.rel
        lines.append(f"## `{rel_display}`")
        lines.append("")
        lines.append(f"- **Type**: `{info.nb_type}`")
        lines.append(f"- **Notebook**: [`{rel_display}`](../{rel_display})")
        lines.append(f"- **Per-notebook doc**: [`{rel_display}.md`](./{Path(rel_display).with_suffix('.md')})")
        lines.append("")

        lines.append("### Goal")
        lines.append("")
        lines.append(info.goal.strip() or "(Not specified)")
        lines.append("")

        lines.append("### Main steps")
        lines.append("")
        for s in (info.steps or ["(No clear section headings found; see notebook for details)"]):
            lines.append(f"- {s}")
        lines.append("")

        outs = relize_paths(info.outputs)
        ins = relize_paths(info.inputs)

        lines.append("### Outputs")
        lines.append("")
        if outs:
            for p in outs[:15]:
                lines.append(f"- `{p}`")
            if len(outs) > 15:
                lines.append(f"- (and {len(outs) - 15} more…)")  # keep file size manageable
        elif info.bq_tables:
            lines.append("- (No obvious file outputs detected; see BigQuery tables below.)")
        else:
            lines.append("- (No obvious outputs detected from static analysis.)")
        lines.append("")

        if info.bq_tables:
            lines.append("### BigQuery tables (referenced)")
            lines.append("")
            for t in info.bq_tables[:15]:
                lines.append(f"- `{t}`")
            if len(info.bq_tables) > 15:
                lines.append(f"- (and {len(info.bq_tables) - 15} more…)")
            lines.append("")

        lines.append("### Suggested ordering (best-effort)")
        lines.append("")
        ra = edges_after.get(info.rel, [])
        rb = edges_before.get(info.rel, [])
        if ra:
            lines.append("- **Runs after**:")
            for x in ra[:10]:
                lines.append(f"  - `{x}`")
        if rb:
            lines.append("- **Runs before**:")
            for x in rb[:10]:
                lines.append(f"  - `{x}`")
        if not ra and not rb:
            lines.append("- (No clear predecessor/successor inferred.)")
        lines.append("")

        # Optional: inputs (helps debugging pipeline)
        lines.append("### Inputs (detected)")
        lines.append("")
        if ins:
            for p in ins[:10]:
                lines.append(f"- `{p}`")
            if len(ins) > 10:
                lines.append(f"- (and {len(ins) - 10} more…)")
        else:
            lines.append("- (No obvious inputs detected.)")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    ap.add_argument("--docs-dir", type=Path, default=Path("notebook_docs"), help="Where to write markdown docs")
    ap.add_argument("--glob", type=str, default="**/*.ipynb", help="Notebook glob")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    docs_root: Path = (root / args.docs_dir).resolve()
    docs_root.mkdir(parents=True, exist_ok=True)

    nb_paths = sorted(root.glob(args.glob))
    # skip ipynb checkpoints
    nb_paths = [p for p in nb_paths if ".ipynb_checkpoints" not in str(p)]

    infos: list[NotebookInfo] = []
    for nb_path in nb_paths:
        rel = _safe_relpath(nb_path, root)
        nb_type = _classify_notebook(rel)
        try:
            nb = _load_notebook(nb_path)
            cells = nb.get("cells") or []
        except Exception:
            # unreadable notebook; still produce stub
            cells = []
        goal = _infer_goal(nb_path, cells)
        steps = _infer_main_steps(cells)
        inputs, outputs, bq_tables = _extract_io(cells, nb_path)
        infos.append(
            NotebookInfo(
                path=nb_path,
                rel=rel,
                nb_type=nb_type,
                goal=goal,
                steps=steps,
                inputs=inputs,
                outputs=outputs,
                bq_tables=bq_tables,
            )
        )

    # Build dependency map from file outputs -> file inputs.
    produced_by: dict[str, list[str]] = {}
    consumed_by: dict[str, list[str]] = {}
    for info in infos:
        for outp in info.outputs:
            produced_by.setdefault(outp, []).append(info.rel)
        for inp in info.inputs:
            consumed_by.setdefault(inp, []).append(info.rel)

    edges_after: dict[str, list[str]] = {info.rel: [] for info in infos}   # nb -> predecessors
    edges_before: dict[str, list[str]] = {info.rel: [] for info in infos}  # nb -> successors

    for path_token, consumers in consumed_by.items():
        producers = produced_by.get(path_token, [])
        if not producers:
            continue
        for c in consumers:
            for p in producers:
                if p == c:
                    continue
                edges_after.setdefault(c, []).append(p)
                edges_before.setdefault(p, []).append(c)

    # de-dup edges and keep stable order
    def dedup_edge_map(m: dict[str, list[str]]) -> dict[str, list[str]]:
        out = {}
        for k, vs in m.items():
            seen = set()
            res: list[str] = []
            for v in vs:
                if v in seen:
                    continue
                seen.add(v)
                res.append(v)
            out[k] = sorted(res, key=_sort_key_for_run_order)
        return out

    edges_after = dedup_edge_map(edges_after)
    edges_before = dedup_edge_map(edges_before)

    # Write per-notebook docs
    for info in infos:
        out_md = docs_root / Path(info.rel).with_suffix(".md")
        out_md.parent.mkdir(parents=True, exist_ok=True)

        ra = [f"`{x}`" for x in edges_after.get(info.rel, [])]
        rb = [f"`{x}`" for x in edges_before.get(info.rel, [])]
        notebook_link_from_doc = os.path.relpath(info.path, out_md.parent).replace("\\", "/")

        md = _render_notebook_md(
            root,
            docs_root,
            info,
            runs_after=ra[:25],
            runs_before=rb[:25],
            notebook_link_from_doc=notebook_link_from_doc,
        )
        out_md.write_text(md, encoding="utf-8")

    # Write index
    infos_sorted = sorted(infos, key=lambda x: _sort_key_for_run_order(x.rel))
    index_md = _render_index_md(root, docs_root, infos_sorted, edges_after, edges_before)
    (docs_root / "INDEX.md").write_text(index_md, encoding="utf-8")

    # Write single consolidated doc
    all_md = _render_all_notebooks_md(root, docs_root, infos_sorted, edges_after, edges_before)
    (docs_root / "ALL_NOTEBOOKS.md").write_text(all_md, encoding="utf-8")

    print(f"Wrote {len(infos)} notebook docs under: {docs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
