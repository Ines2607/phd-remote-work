"""Monthly data-quality statistics of raw dwells and outlier months.

Ports ``demo/001-generate-analyse-dwells/demo-collect-dwells-statistic.py``,
with weekends / holidays from the Israeli calendar in the settings. Parameters
are in the ``dwell_stats`` settings block.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from remote_work_utils import workdays
from remote_work_utils.geo import encode_geohash


def _period_labels(hours: pd.Series, params: dict[str, Any]) -> pd.Series:
    labels = pd.Series(params["other_time_period"], index=hours.index, dtype="object")
    for window in params["time_periods"]:
        labels[(hours >= window["start_hour"]) & (hours < window["end_hour"])] = window["name"]
    return labels


def monthly_dwell_stats(raw: pd.DataFrame, month: str, settings: dict[str, Any]) -> dict[str, Any]:
    """One row of statistics for a month of raw dwells (raw input schema)."""
    params = settings["dwell_stats"]
    local = pd.to_datetime(raw["local_date_time"])
    dates = local.dt.normalize()
    hours = raw["duration_seconds"] / 3600
    geohash = encode_geohash(
        raw["centroid_latitude"],
        raw["centroid_longitude"],
        settings["geohash"]["precision"],
        settings["geohash"]["lonlat_swapped"],
    )
    non_workday = workdays.flag_non_workday(dates, settings["calendar"]).to_numpy()
    frame = pd.DataFrame(
        {
            "identifier": raw["identifier"].to_numpy(),
            "identifier_type": raw["identifier_type"].to_numpy(),
            "date": dates.to_numpy(),
            "geohash": np.asarray(geohash),
            "hours": hours.to_numpy(),
            "non_workday": non_workday,
        }
    )
    users = frame.groupby(["identifier", "identifier_type"]).agg(
        dwells=("date", "size"),
        days=("date", "nunique"),
        geohashes=("geohash", "nunique"),
        total_hours=("hours", "sum"),
    )
    users["workdays"] = frame[~frame["non_workday"]].groupby("identifier")["date"].nunique().reindex(
        users.index.get_level_values("identifier")
    ).fillna(0).to_numpy()
    users = users.reset_index()

    n_users = len(users)
    n_dwells = len(frame)
    stats: dict[str, Any] = {"month": month, "total_users": n_users, "total_dwells": n_dwells}
    for kind, count in users["identifier_type"].value_counts().items():
        stats[f"share_{str(kind).lower()}"] = count / n_users
    for q in params["dwell_quantiles"]:
        stats[f"p{int(q * 100)}_dwells_per_user"] = users["dwells"].quantile(q)
    bins = params["dwell_count_bins"]
    for low, high in zip(bins, bins[1:] + [None]):
        mask = users["dwells"] >= low if high is None else users["dwells"].between(low, high - 1)
        label = f"{low}plus" if high is None else str(low) if high - 1 == low else f"{low}_{high - 1}"
        stats[f"share_users_{label}_dwells"] = mask.mean()
    stats.update(
        {
            "median_days_per_user": users["days"].median(),
            "median_workdays_per_user": users["workdays"].median(),
            "median_geohashes_per_user": users["geohashes"].median(),
            "median_total_hours_per_user": users["total_hours"].median(),
            "dwells_per_day_per_user": (users["dwells"] / users["days"]).mean(),
            "median_duration_hours": frame["hours"].median(),
            "share_dwells_non_workday": frame["non_workday"].mean(),
            "share_dwells_over_min_duration": (raw["duration_seconds"] > params["min_duration_seconds"]).mean(),
            "total_unique_geohashes": frame["geohash"].nunique(),
        }
    )
    for threshold in params["long_dwell_hours"]:
        stats[f"share_dwells_over_{threshold}h"] = (frame["hours"] > threshold).mean()
    periods = _period_labels(local.dt.hour, params).value_counts(normalize=True)
    for name in [w["name"] for w in params["time_periods"]] + [params["other_time_period"]]:
        stats[f"share_dwells_{name}"] = periods.get(name, 0.0)
    return stats


def outlier_months(monthly: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Months that are unusually low on many statistics.

    A statistic is an outlier for a month when it is more than ``n_std``
    standard deviations below the median across months, a count equal to 0,
    or a share equal to 1. Months above ``remove_outlier_pct`` percent of
    outlier statistics are flagged for removal, above ``warning_outlier_pct``
    for a warning.
    """
    columns = [c for c in monthly.select_dtypes(include=[np.number]).columns if c != "month"]
    flags = pd.DataFrame(index=monthly.index)
    for column in columns:
        values = monthly[column]
        low = values < values.median() - params["n_std"] * values.std()
        degenerate = values.eq(1) if column.startswith("share") else values.eq(0)
        flags[column] = low | degenerate
    percent = 100 * flags.sum(axis=1) / len(columns)
    return pd.DataFrame(
        {
            "month": monthly["month"],
            "outlier_statistics": flags.sum(axis=1),
            "outlier_percent": percent.round(2),
            "remove": percent > params["remove_outlier_pct"],
            "warning": percent.between(params["warning_outlier_pct"], params["remove_outlier_pct"]),
            "outlier_columns": flags.apply(lambda row: ", ".join(row.index[row]), axis=1),
        }
    ).sort_values("month", ignore_index=True)
