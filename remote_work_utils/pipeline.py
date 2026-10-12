"""Run the monthly steps end to end and write outputs.

Steps: ``prepare`` (raw -> filtered dwells), ``home``, ``work``, ``remote``.
Output file names come from ``settings["outputs"]`` relative to
``paths.output_dir``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from remote_work_utils.home_work.home import detect_homes, homes_by_stat_area
from remote_work_utils.home_work.work import detect_work
from remote_work_utils.prepare import prepare_dwells
from remote_work_utils.remote_work.hourly import build_hourly_panel
from remote_work_utils.remote_work.metrics import hour_date_stats, remote_work_month
from remote_work_utils.schema import read_dwells
from remote_work_utils.settings import resolve_path

STEPS = ["prepare", "home", "work", "remote"]


@dataclass
class MonthResult:
    month: str
    homes: pd.DataFrame | None = None
    users_work_home: pd.DataFrame | None = None
    user_remote_work: pd.DataFrame | None = None
    hours: pd.DataFrame | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def output_path(settings: dict[str, Any], name: str, month: str) -> Path:
    path = resolve_path(settings, "output_dir") / settings["outputs"][name].format(month=month)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_prepare(settings: dict[str, Any], month: str) -> pd.DataFrame:
    raw = pd.read_parquet(resolve_path(settings, "dwells_raw", month=month))
    boundary = gpd.read_file(resolve_path(settings, "boundary")) if settings["prepare"]["clip_to_boundary"] else None
    filtered = prepare_dwells(raw, settings, boundary)
    target = resolve_path(settings, "dwells", month=month)
    target.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(target, index=False)
    return filtered


def _read_csv(settings: dict[str, Any], name: str, month: str) -> pd.DataFrame:
    return pd.read_csv(output_path(settings, name, month))


def run_month(settings: dict[str, Any], month: str, steps: list[str] | None = None, write: bool = True) -> MonthResult:
    """Run the selected steps for one month (``YYYYMM``)."""
    steps = steps or STEPS[1:]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown steps {sorted(unknown)}; choose from {STEPS}")
    result = MonthResult(month=month)
    if "prepare" in steps:
        run_prepare(settings, month)
    if not {"home", "work", "remote"} & set(steps):
        return result

    dwells = read_dwells(resolve_path(settings, "dwells", month=month), settings)

    if "home" in steps:
        home = detect_homes(dwells, settings)
        if home.homes.empty:
            raise ValueError(f"No home locations identified for {month}")
        result.homes = home.homes
        result.metrics.update(home.metrics)
        stat_areas_path = resolve_path(settings, "stat_areas")
        if stat_areas_path.exists():
            areas, correlation = homes_by_stat_area(home.homes, gpd.read_file(stat_areas_path), settings)
            result.metrics["home_pop_corr"] = correlation
            if write:
                areas.to_file(output_path(settings, "home_stat_areas", month))
        if write:
            home.homes.to_csv(output_path(settings, "homes", month), index=False)
    elif {"work", "remote"} & set(steps):
        result.homes = _read_csv(settings, "homes", month)

    if "work" in steps:
        result.users_work_home = detect_work(dwells, result.homes, settings)
        result.metrics["num_works"] = int(len(result.users_work_home))
        if write:
            result.users_work_home.to_csv(output_path(settings, "work", month), index=False)
    elif "remote" in steps:
        result.users_work_home = _read_csv(settings, "work", month)

    if "remote" in steps:
        hours = build_hourly_panel(dwells, result.users_work_home, settings, month)
        result.user_remote_work, result.hours = remote_work_month(hours, settings)
        result.metrics["num_remote_work_users"] = int(len(result.user_remote_work))
        if write:
            result.user_remote_work.to_csv(output_path(settings, "remote_work_users", month), index=False)
            hour_date_stats(result.hours).to_csv(output_path(settings, "remote_work_hour_stats", month), index=False)

    if write:
        with output_path(settings, "metrics_log", month).open("a", encoding="utf-8") as log:
            log.write(json.dumps({"month": month, **result.metrics}, default=float) + "\n")
    return result
