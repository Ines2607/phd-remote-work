"""POI visit statistics and rule-based remote-work suitability flags.

Ports ``800-research-users-locations/005-third-places-users-signals`` (visit
statistics) and ``060-process-pois/103-google-pois-flags-pipeline.sql``
(category, location and user-behaviour flags).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _label_windows(hours: pd.Series, windows: list[dict[str, Any]], other: str) -> pd.Series:
    labels = pd.Series(other, index=hours.index, dtype="object")
    for window in windows:
        labels[(hours >= window["start_hour"]) & (hours < window["end_hour"])] = window["name"]
    return labels


def _label_days(dates: pd.Series, day_types: dict[str, list[int]]) -> pd.Series:
    weekday = pd.to_datetime(dates).dt.weekday
    labels = pd.Series(None, index=dates.index, dtype="object")
    for name, weekdays in day_types.items():
        labels[weekday.isin(weekdays)] = name
    return labels


def poi_visit_stats(visits: pd.DataFrame, params: dict[str, Any], group: list[str] | None = None) -> pd.DataFrame:
    """Per-POI visit metrics (optionally per extra ``group`` columns, e.g. month, worker type).

    ``visits``: non-home, non-work dwells with ``poi_id``, ``identifier``,
    ``start_date_time``, ``date``, ``duration_hours``, ``dist_home``,
    ``dist_work``, ``after_home``, ``after_work``, ``prev_flag_work``.
    """
    v = visits[visits["poi_id"].notna()].copy()
    v["time_period"] = _label_windows(v["start_date_time"].dt.hour, params["time_periods"], params["other_time_period"])
    v["day_type"] = _label_days(v["date"], params["day_types"])
    weekday = v["day_type"] == "weekday"
    long_visit = v["duration_hours"] > params["long_visit_hours"]
    rw_period = v["time_period"].isin(params["remote_work_time_periods"])
    near = params["near_distance_m"]
    v = v.assign(
        _weekday=weekday,
        _long_weekday=long_visit & weekday,
        _long_morning_weekday=long_visit & weekday & rw_period,
        _remote_work_visit=long_visit & weekday & rw_period & ~v["prev_flag_work"],
        _friday=v["day_type"] == "friday",
        _saturday=v["day_type"] == "saturday",
        _home_near=v["dist_home"] < near,
        _work_near=v["dist_work"] < near,
        _duration_weekday=v["duration_hours"].where(weekday),
        **{f"_period_{w['name']}": v["time_period"] == w["name"] for w in params["time_periods"]},
    )
    keys = ["poi_id", *(group or [])]
    grouped = v.groupby(keys)
    stats = grouped.agg(
        n_visits=("identifier", "size"),
        n_unique_users=("identifier", "nunique"),
        n_visits_weekday=("_weekday", "sum"),
        n_long_visits_weekday=("_long_weekday", "sum"),
        n_long_morning_weekday_visits=("_long_morning_weekday", "sum"),
        n_remote_work_visits=("_remote_work_visit", "sum"),
        n_visits_after_home=("after_home", "sum"),
        n_visits_after_work=("after_work", "sum"),
        avg_duration_hours=("duration_hours", "mean"),
        avg_duration_hours_weekday=("_duration_weekday", "mean"),
        share_weekday=("_weekday", "mean"),
        share_friday=("_friday", "mean"),
        share_saturday=("_saturday", "mean"),
        avg_dist_home=("dist_home", "mean"),
        avg_dist_work=("dist_work", "mean"),
        share_dwells_home_lt_1km=("_home_near", "mean"),
        share_dwells_work_lt_1km=("_work_near", "mean"),
        **{f"share_{w['name']}": (f"_period_{w['name']}", "mean") for w in params["time_periods"]},
    )
    stats["visit_frequency"] = stats["n_visits"] / stats["n_unique_users"]
    stats["share_visits_after_home"] = stats["n_visits_after_home"] / stats["n_visits"]
    stats["share_visits_after_work"] = stats["n_visits_after_work"] / stats["n_visits"]
    stats["rw_share"] = stats["n_long_morning_weekday_visits"] / stats["n_visits"]
    return stats[stats["n_visits"] >= params["min_visits"]].reset_index()


def flag_category(categories: pd.Series, params: dict[str, Any]) -> pd.Series:
    """True for remote-work-friendly categories, False for office/institution ones, else missing."""
    lowered = categories.str.lower()
    flag = pd.Series(pd.NA, index=categories.index, dtype="boolean")
    flag[lowered.isin(params["category_suits_rw"])] = True
    flag[lowered.isin(params["category_not_suits_rw"])] = False
    return flag


def flag_location(geohash_features: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """True where the POI's cell has no universities, schools, hospitals, hubs and is not a work zone."""
    institutions = geohash_features[params["location_exclusion_counts"]].fillna(0).sum(axis=1)
    return (institutions == 0) & (geohash_features[params["work_zone_column"]].fillna(0) == 0)


def flag_user_behaviour(remote_worker_stats: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """True when remote workers' long weekday late-morning visits are a large enough share."""
    return pd.Series(
        np.where(
            remote_worker_stats["n_visits_weekday"] == 0,
            False,
            remote_worker_stats["rw_share"] >= params["behaviour_rw_share_threshold"],
        ),
        index=remote_worker_stats.index,
    )
