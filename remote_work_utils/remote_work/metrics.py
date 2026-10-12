"""Remote-work flags and the user-level monthly metric.

Port of ``process_month_remote_work`` in
``020-process-dwells/003-remote-work-monthly.py``. The paper metric is
``flag_remote_work_smoothed_max_thresh``: the share of a user's working days
(days with at least one predicted work hour) on which some hour has
``flag_remote_work * P_general(work | hour) > smoothed_day_threshold``.
See ``notebook_docs/methodology/REMOTE_WORK_FLAGS.md``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from remote_work_utils.remote_work.hourly import PROBABILITY_COLUMNS

SELECTED_METRIC = "flag_remote_work_smoothed_max_thresh"
CORE_USER_COLUMNS = ["identifier", "identifier_type", "total_work_days", "office_day_counts", SELECTED_METRIC]


def general_location_shares(hours: pd.DataFrame) -> pd.DataFrame:
    """Share of all users' office-day hours at home / work / other, by clock hour."""
    office = hours[(hours["office_day"]) & (~hours["flag_weekend"])]
    if office.empty:
        raise ValueError("No office-day hours found")
    return office.groupby(["hour", "location_type"]).size().unstack(fill_value=0).div(
        office.groupby("hour").size(), axis=0
    )


def add_remote_work_flags(hours: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Hour-level predictions and the remote-work flag variants."""
    hours = hours.copy()
    hours["prob_3rdplace_at_hour"] = 1 - (hours["prob_office_at_hour"] + hours["prob_home_at_hour"])
    hours = hours.merge(general_location_shares(hours).reset_index(), on="hour", how="left").fillna(0)
    missing = [column for column in ["work", "home", "other"] if column not in hours.columns]
    if missing:
        raise ValueError(f"No office-day hours with location type(s) {missing}")

    hours["predicted_location_type"] = (
        hours[PROBABILITY_COLUMNS]
        .idxmax(axis=1)
        .map({"prob_home_at_hour": "home", "prob_office_at_hour": "work", "prob_3rdplace_at_hour": "other"})
    )
    hours["is_working_hours"] = hours["hour"].between(params["working_hour_start"], params["working_hour_end"])
    hours["flag_work_hours"] = (hours["predicted_location_type"] == "work") & hours["is_working_hours"]
    hours["flag_work_hours_day"] = hours.groupby(["identifier", "date"])["flag_work_hours"].transform("sum") > 0

    at_home_remote = (hours["location_type"] == "home") & hours["is_working_hours"] & (~hours["office_day"])
    hours["flag_remote_work"] = at_home_remote & hours["flag_work_hours"]

    hours["smoothed_prob_office"] = hours["prob_office_at_hour"] * hours["work"]
    hours["smoothed_prob_home"] = hours["prob_home_at_hour"] * hours["home"]
    hours["smoothed_prob_3rdplace"] = hours["prob_3rdplace_at_hour"] * hours["other"]
    hours["predicted_location_type_smooth_prob"] = hours[
        ["smoothed_prob_home", "smoothed_prob_office", "smoothed_prob_3rdplace"]
    ].idxmax(axis=1)
    hours["flag_remote_work_smooth_prob"] = at_home_remote & (
        hours["predicted_location_type_smooth_prob"] == "smoothed_prob_office"
    )
    hours["flag_remote_work_min_work_prob"] = hours["flag_remote_work"] & (
        hours["work"] > params["min_work_probability"]
    )
    hours["flag_remote_work_smoothed"] = hours["flag_remote_work"] * hours["work"]
    hours["remote_work_smooth_prob"] = (
        hours["smoothed_prob_office"] * hours["flag_home_geohash"] * (~hours["office_day"])
    )
    return hours


def user_remote_work(hours: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """One row per user: remote-work days, working days and the paper metric."""
    work_days = hours[hours["flag_work_hours_day"]]
    day_flags = (
        work_days.groupby(["identifier", "identifier_type", "date"])[
            ["flag_remote_work", "flag_remote_work_smooth_prob", "flag_remote_work_min_work_prob"]
        ]
        .sum()
        .ge(1)
        .reset_index()
    )
    users = (
        day_flags.groupby(["identifier", "identifier_type"])
        .agg(
            {
                "flag_remote_work": ["sum", "count"],
                "flag_remote_work_smooth_prob": "sum",
                "flag_remote_work_min_work_prob": "sum",
            }
        )
        .reset_index()
    )
    users.columns = [
        "identifier",
        "identifier_type",
        "remote_work_days",
        "total_work_days",
        "remote_work_days_smooth",
        "remote_work_days_min_work",
    ]
    daily_max = work_days.groupby(["identifier", "date"])["flag_remote_work_smoothed"].max()
    users = users.merge(
        daily_max.groupby("identifier").mean().rename("flag_remote_work_smoothed_max"), on="identifier", how="left"
    ).merge(
        (daily_max > params["smoothed_day_threshold"]).groupby("identifier").mean().rename(SELECTED_METRIC),
        on="identifier",
        how="left",
    )
    for suffix in ["", "_smooth", "_min_work"]:
        users[f"remote_work_share{suffix}"] = users[f"remote_work_days{suffix}"] / users["total_work_days"]
    users = users.merge(
        hours[hours["office_day"]].groupby("identifier")["date"].nunique().rename("office_day_counts").to_frame(),
        on="identifier",
        how="left",
    )
    if not params["output_all_variants"]:
        users = users[CORE_USER_COLUMNS]
    return users


def hour_date_stats(hours: pd.DataFrame) -> pd.DataFrame:
    """Counts of remote-work and work hours by date, hour and device type."""
    keys = ["date", "hour", "identifier_type"]
    return (
        hours.groupby(keys)[["flag_remote_work", "flag_work_hours", "flag_remote_work_smoothed"]]
        .sum()
        .join(hours.groupby(keys).size().rename("count_dwells"))
        .reset_index()
    )


def remote_work_month(hours: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flags on the hourly panel and the user table for one month."""
    params = settings["remote_work"]
    flagged = add_remote_work_flags(hours, params)
    return user_remote_work(flagged, params), flagged
