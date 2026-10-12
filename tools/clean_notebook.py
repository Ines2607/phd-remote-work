"""Make cleaned copies of curated research notebooks.

    python tools/clean_notebook.py                     # every notebook in research/MANIFEST.yml
    python tools/clean_notebook.py --only 3000-pu      # entries whose target contains the text

For each manifest entry the source notebook (left untouched) is copied to
``target`` without outputs and without:

* **checks** — cells that only inspect data (``.head()``, ``.shape``,
  ``.value_counts()``, ``print(len(df))``, asserts, bare variable displays);
  trailing check lines are also trimmed from cells that do real work;
* **empty / commented-out cells**;
* **dead ends** — cells whose names no later kept cell uses and that write,
  show or plot nothing, or that raised an error in the original run;
* **unlabelled plots** — figures without title or axis labels that are not saved;
* outputs, unless the entry sets ``keep_outputs: true`` (then only error
  outputs and outputs showing device identifiers are removed);
* cells listed under ``drop_cells`` or matching ``drop_patterns`` (with the
  reason given in the manifest), e.g. rent / housing-price cells.

Dependency safety: cells are visited from the last to the first, and a
candidate is removed only if none of the names it defines is used by a later
kept cell. Cells dropped on purpose that feed a kept cell are reported as
``BROKEN`` so the manifest can be fixed. Remaining matches of
``forbidden_patterns`` (secrets, absolute paths, excluded topics) are reported
for attention. The report is written to ``research/CLEANING_REPORT.md``.

The manifest and the report stay local (gitignored): the source notebooks are
not published, and both files name the content that was removed.
"""

from __future__ import annotations

import argparse
import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research" / "MANIFEST.yml"
REPORT = ROOT / "research" / "CLEANING_REPORT.md"

CHECK_METHODS = {
    "head", "tail", "info", "describe", "value_counts", "sample", "nunique", "unique", "isna", "isnull",
    "notna", "duplicated", "memory_usage", "count", "dtypes", "shape", "columns", "keys", "max", "min",
    "sum", "mean", "median", "std", "quantile",
}
CHECK_ATTRIBUTES = {"shape", "dtypes", "columns", "index", "size", "crs", "total_bounds", "empty"}
BUILTIN_CALLS = {
    "len", "type", "list", "dict", "set", "tuple", "sorted", "str", "int", "float", "range", "sum", "max", "min",
    "print", "display", "zip", "enumerate", "isinstance", "getattr",
}
SQL_DDL = re.compile(r"\b(CREATE|INSERT|MERGE|UPDATE)\b", re.IGNORECASE)
MUTATING_METHODS = {"append", "extend", "update", "insert", "pop", "remove", "add", "setdefault", "clear"}
OUTPUT_PATTERN = re.compile(
    r"\.to_(csv|parquet|file|excel|gbq|json|pickle|sql)\(|savefig|write_|\.write\(|dump\(|upload|load_table|"
    r"read_query|\.query\(|execute|CREATE|INSERT|DELETE|MERGE|mkdir|print\(|display\(|\.show\(|\.plot|plt\.|sns\.|"
    r"px\.|go\.|folium|\.explore\(|summary\(|Map\(|imshow|hist\(",
    re.IGNORECASE,
)
SETUP_PATTERN = re.compile(r"sys\.path|os\.environ|set_option|filterwarnings|set_theme|rcParams")
PLOT_PATTERN = re.compile(r"\.plot\(|\.plot\.|plt\.|\bsns?\.|px\.|\.hist\(|\.bar\(|\.scatter\(|\.explore\(|folium", re.IGNORECASE)
LABEL_PATTERN = re.compile(r"title|xlabel|ylabel|savefig|labels\s*=|legend", re.IGNORECASE)
MAGIC = re.compile(r"^\s*([%!?])")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


@dataclass
class CellReport:
    index: int
    first_line: str
    reason: str


@dataclass
class NotebookReport:
    source: str
    target: str
    kept: int = 0
    removed: list[CellReport] = field(default_factory=list)
    trimmed: list[CellReport] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    attention: list[str] = field(default_factory=list)
    unparsed: list[int] = field(default_factory=list)
    scrubbed: list[int] = field(default_factory=list)


def _python_source(source: str) -> tuple[str, set[str]]:
    """Replace IPython magics with ``pass``; ``%%bigquery name`` defines ``name``."""
    defines = set()
    lines = source.splitlines()
    if lines and lines[0].startswith("%%"):
        parts = lines[0].split()
        if parts[0] == "%%bigquery" and len(parts) > 1 and not parts[1].startswith("-"):
            defines.add(parts[1])
            return "pass", defines
        if parts[0] in {"%%time", "%%timeit", "%%capture"}:
            lines = lines[1:]
        else:
            return "pass", defines
    return textwrap.dedent("\n".join("pass" if MAGIC.match(line) else line for line in lines)), defines


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _names(tree: ast.AST) -> tuple[set[str], set[str]]:
    defines, uses = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (defines if isinstance(node.ctx, (ast.Store, ast.Del)) else uses).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defines.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defines.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, (ast.Subscript, ast.Attribute)) and (name := _root_name(target)):
                    defines.add(name)
                    uses.add(name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            inplace = any(k.arg == "inplace" for k in node.keywords)
            if node.func.attr in MUTATING_METHODS or inplace:
                if name := _root_name(node.func.value):
                    defines.add(name)
    return defines, uses


def _is_check(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Assert):
        return True
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str) and SQL_DDL.search(value.value):
        return False
    if isinstance(value, (ast.Name, ast.Subscript, ast.Constant)):
        return True
    if isinstance(value, ast.Attribute):
        return value.attr in CHECK_ATTRIBUTES
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Attribute) and func.attr in CHECK_METHODS:
            return True
        if isinstance(func, ast.Name) and func.id in {"len", "type"}:
            return True
        if isinstance(func, ast.Name) and func.id in {"print", "display"}:
            return all(_is_check(ast.Expr(value=arg)) or isinstance(arg, ast.JoinedStr) for arg in value.args) and all(
                not isinstance(arg, ast.JoinedStr) or _fstring_is_check(arg) for arg in value.args
            )
    return False


def _fstring_is_check(node: ast.JoinedStr) -> bool:
    values = [v.value for v in node.values if isinstance(v, ast.FormattedValue)]
    return bool(values) and all(_is_check(ast.Expr(value=v)) for v in values)


def _first_line(source: str) -> str:
    for line in source.splitlines():
        if line.strip():
            return line.strip()[:90].replace("|", "\\|")
    return "(empty)"


def _is_heading(cell) -> bool:
    lines = [line for line in cell.source.splitlines() if line.strip()]
    return cell.cell_type == "markdown" and (not lines or (len(lines) == 1 and lines[0].lstrip().startswith("#")))


def _had_error(cell) -> bool:
    return any(output.get("output_type") == "error" and output.get("ename") for output in cell.get("outputs", []))


def _has_image(cell) -> bool:
    return any("image/png" in output.get("data", {}) for output in cell.get("outputs", []))


def _calls_function(tree: ast.Module) -> bool:
    """A top-level statement calls a plain function (helpers that print, plot or fit models)."""
    for statement in tree.body:
        value = statement.value if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id not in BUILTIN_CALLS:
            return True
    return False


def _trim_checks(source: str, tree: ast.Module) -> str:
    """Drop trailing top-level check statements from a cell that does other work."""
    body = list(tree.body)
    lines = source.splitlines()
    end = len(lines)
    while len(body) > 1 and _is_check(body[-1]):
        end = body[-1].lineno - 1
        body.pop()
    return "\n".join(lines[:end]).rstrip()


def clean(entry: dict, defaults: dict) -> tuple[nbformat.NotebookNode, NotebookReport]:
    source_path = ROOT / entry["source"]
    notebook = nbformat.read(source_path, as_version=4)
    report = NotebookReport(entry["source"], entry["target"])
    drop_cells = {int(k): v for k, v in (entry.get("drop_cells") or {}).items()}
    keep_cells = set(entry.get("keep_cells") or [])
    drop_patterns = [(re.compile(p["pattern"], re.IGNORECASE), p["reason"]) for p in entry.get("drop_patterns") or []]
    for replacement in entry.get("replace") or []:
        cell = notebook.cells[replacement["cell"]]
        if replacement["old"] not in cell.source:
            report.broken.append(f"cell {replacement['cell']}: replacement text not found: {replacement['old'][:60]!r}")
        cell.source = cell.source.replace(replacement["old"], replacement["new"])
    root = "/".join([".."] * len(Path(entry["target"]).parent.parts))
    for replacement in defaults.get("replace_all") or []:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace(replacement["old"], replacement["new"].format(root=root))

    parsed = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code, magic_defines = _python_source(cell.source)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            report.unparsed.append(index)
            tree = None
        defines, uses = _names(tree) if tree else (set(), set())
        parsed[index] = (tree, defines | magic_defines, uses)

    used_later: set[str] = set()
    removed: set[int] = set()
    for index, reason in drop_cells.items():
        if notebook.cells[index].cell_type != "code":
            removed.add(index)
            report.removed.append(CellReport(index, _first_line(notebook.cells[index].source), reason))
    for index in sorted(parsed, reverse=True):
        cell = notebook.cells[index]
        tree, defines, uses = parsed[index]
        stripped = [line for line in cell.source.splitlines() if line.strip()]
        reason = None
        forced = False
        if index in drop_cells:
            reason, forced = drop_cells[index], True
        elif any(pattern.search(cell.source) for pattern, _ in drop_patterns):
            reason = next(r for p, r in drop_patterns if p.search(cell.source))
            forced = True
        elif index in keep_cells or tree is None:
            reason = None
        elif not stripped:
            reason = "empty"
        elif all(line.lstrip().startswith("#") for line in stripped):
            reason = "commented-out code"
        elif tree.body and all(_is_check(statement) for statement in tree.body):
            reason = "check (inspects data only)"
        elif _had_error(cell) and not (defines & used_later):
            reason = "raised an error in the original run"
        elif PLOT_PATTERN.search(cell.source) and not LABEL_PATTERN.search(cell.source) and not (defines & used_later):
            reason = "plot without title or labels"
        elif (
            not OUTPUT_PATTERN.search(cell.source)
            and not SETUP_PATTERN.search(cell.source)
            and not (defines & used_later)
            and not _calls_function(tree)
            and not _has_image(cell)
        ):
            reason = "dead end (nothing later uses its results)"

        if reason and forced:
            needed = defines & used_later
            if needed:
                report.broken.append(f"cell {index} ({reason}) defines {sorted(needed)} used by later kept cells")
            removed.add(index)
            report.removed.append(CellReport(index, _first_line(cell.source), reason))
            continue
        if reason and not (defines & used_later):
            removed.add(index)
            report.removed.append(CellReport(index, _first_line(cell.source), reason))
            continue
        if tree is not None and len(tree.body) > 1 and _is_check(tree.body[-1]) and not cell.source.startswith("%"):
            trimmed = _trim_checks(cell.source, tree)
            if trimmed and trimmed != cell.source.rstrip():
                report.trimmed.append(CellReport(index, _first_line(cell.source), "trailing check lines removed"))
                cell.source = trimmed
        used_later |= uses

    kept_indices = [i for i in range(len(notebook.cells)) if i not in removed]
    for position, index in enumerate(kept_indices):
        cell = notebook.cells[index]
        following = notebook.cells[kept_indices[position + 1]] if position + 1 < len(kept_indices) else None
        if _is_heading(cell) and (following is None or _is_heading(following)):
            removed.add(index)
            report.removed.append(CellReport(index, _first_line(cell.source), "heading of a removed section"))

    keep_outputs = bool(entry.get("keep_outputs"))
    cells = []
    for index, cell in enumerate(notebook.cells):
        if index in removed:
            continue
        if cell.cell_type == "code":
            if keep_outputs:
                kept = [o for o in cell.outputs if o.get("output_type") != "error" and not UUID.search(str(o))]
                if len(kept) < len(cell.outputs):
                    report.scrubbed.append(index)
                cell.outputs = kept
            else:
                cell.outputs = []
                cell.execution_count = None
        cell.metadata = {}
        cells.append(cell)
    report.removed.sort(key=lambda r: r.index)
    report.trimmed.sort(key=lambda r: r.index)
    report.kept = sum(c.cell_type == "code" for c in cells)

    outputs_note = "outputs from the original run kept" if keep_outputs else "outputs cleared"
    header = nbformat.v4.new_markdown_cell(
        f"> **Research record.** Cleaned copy of `{entry['source']}` "
        f"(data checks, dead ends and unlabelled plots removed; {outputs_note}).\n"
        f"> {entry['purpose'].strip()}\n>\n> {defaults['requirements'].strip()}"
    )
    notebook.cells = [header, *cells]
    notebook.metadata = {k: v for k, v in notebook.metadata.items() if k in {"kernelspec", "language_info"}}

    patterns = [*defaults.get("forbidden_patterns", []), *(entry.get("forbidden_patterns") or [])]
    for position, cell in enumerate(notebook.cells[1:], start=1):
        for item in patterns:
            for match in re.finditer(item["pattern"], cell.source, re.IGNORECASE):
                line = cell.source[: match.start()].count("\n")
                text = cell.source.splitlines()[line].strip()[:100].replace("|", "\\|")
                report.attention.append(f"copy cell {position}: {item['reason']}: `{text}`")
                break
    return notebook, report


def write_report(reports: list[NotebookReport]) -> None:
    lines = [
        "# Cleaning report",
        "",
        "Generated by `tools/clean_notebook.py` from `research/MANIFEST.yml`. Cell numbers refer to the",
        "**source** notebook (0-based), except under *Attention*, which refers to the cleaned copy.",
        "",
        "| Copy | Code cells kept | Cells removed | Cells trimmed | Broken | Attention |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| `{r.target}` | {r.kept} | {len(r.removed)} | {len(r.trimmed)} | {len(r.broken)} | {len(r.attention)} |"
        )
    for r in reports:
        lines += ["", f"## `{r.target}`", "", f"Source: `{r.source}`", ""]
        if r.broken:
            lines += ["**BROKEN — fix the manifest:**", *[f"- {b}" for b in r.broken], ""]
        if r.attention:
            lines += ["**Attention:**", *[f"- {a}" for a in r.attention], ""]
        if r.unparsed:
            lines += [f"Not parsed (kept as is): cells {r.unparsed}", ""]
        if r.scrubbed:
            lines += [f"Outputs removed (error or device identifier): cells {r.scrubbed}", ""]
        if r.removed:
            lines += ["| Cell | First line | Reason |", "|---|---|---|"]
            lines += [f"| {c.index} | `{c.first_line}` | {c.reason} |" for c in r.removed]
        if r.trimmed:
            lines += ["", "Trimmed: " + ", ".join(str(c.index) for c in r.trimmed)]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="Process manifest entries whose target contains this text")
    args = parser.parse_args()
    manifest = yaml.safe_load(MANIFEST.read_text())
    reports = []
    for entry in manifest["notebooks"]:
        if args.only and args.only not in entry["target"]:
            continue
        notebook, report = clean(entry, manifest["defaults"])
        target = ROOT / entry["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
        reports.append(report)
        print(f"{entry['target']}: kept {report.kept}, removed {len(report.removed)}, broken {len(report.broken)}")
    if not args.only:
        write_report(reports)


if __name__ == "__main__":
    main()
