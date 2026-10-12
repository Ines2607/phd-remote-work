"""Load region settings (JSON) and resolve data paths.

A settings file may set ``"extends": "<other file>.json"``; the parent is loaded
first (relative to the child) and the child overrides it key by key.

Path values may contain ``{raw}`` / ``{processed}`` placeholders, filled from
the ``[Paths]`` section of a local ``settings.ini`` (see
``settings.example.ini``), and ``{month}`` / ``{year}`` placeholders, filled
when a path is resolved for a specific month. Relative paths are resolved
against the directory of the settings file that defines them.
"""

from __future__ import annotations

import configparser
import copy
import json
import os
from pathlib import Path
from typing import Any

INI_ENV_VAR = "REMOTE_WORK_SETTINGS_INI"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _anchor_paths(paths: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Make relative path entries absolute with respect to their config file."""
    anchored = {}
    for key, value in paths.items():
        if isinstance(value, str) and not value.startswith("{") and not Path(value).is_absolute():
            anchored[key] = str((config_dir / value).resolve())
        else:
            anchored[key] = value
    return anchored


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    parent = data.pop("extends", None)
    if "paths" in data:
        data["paths"] = _anchor_paths(data["paths"], path.parent)
    if parent:
        data = _deep_merge(_read_json((path.parent / parent).resolve()), data)
    return data


def _find_ini(start: Path) -> Path | None:
    if os.environ.get(INI_ENV_VAR):
        return Path(os.environ[INI_ENV_VAR])
    for folder in [start, *start.parents, Path.cwd(), *Path.cwd().parents]:
        candidate = folder / "settings.ini"
        if candidate.exists():
            return candidate
    return None


def _ini_roots(ini_path: Path | None) -> dict[str, str]:
    if ini_path is None:
        return {}
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    if not parser.has_section("Paths"):
        return {}
    return {key: value for key, value in parser.items("Paths")}


def load_settings(path: str | Path) -> dict[str, Any]:
    """Load a settings JSON (following ``extends``) and expand path roots."""
    path = Path(path).resolve()
    settings = _read_json(path)
    roots = _ini_roots(_find_ini(path.parent))
    paths = settings.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, str):
            for root_name, root_value in roots.items():
                value = value.replace("{" + root_name + "}", root_value)
            paths[key] = value
    settings["paths"] = paths
    settings["_source"] = str(path)
    return settings


def resolve_path(settings: dict[str, Any], key: str, **placeholders: str) -> Path:
    """Return ``settings["paths"][key]`` with ``{month}``-style placeholders filled."""
    try:
        template = settings["paths"][key]
    except KeyError as exc:
        raise KeyError(f"paths.{key} is not defined in {settings.get('_source')}") from exc
    if "month" in placeholders and "year" not in placeholders:
        placeholders["year"] = str(placeholders["month"])[:4]
    value = template.format(**placeholders)
    if "{" in value:
        raise ValueError(
            f"paths.{key} still has unresolved placeholders: {value}. "
            "Define the roots in settings.ini (see settings.example.ini)."
        )
    return Path(value)


def section(settings: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required top-level settings section."""
    if name not in settings:
        raise KeyError(f"Settings section '{name}' is missing in {settings.get('_source')}")
    return settings[name]
