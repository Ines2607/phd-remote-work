"""Sentences (visit sessions) for place2vec training.

Same steps as the paper corpus: keep visits of at least ``min_duration_hours``
on workdays, split each user's day into sessions at gaps longer than
``session_gap_hours``, tokenise, drop rare POIs (``min_poi_count``) and
collapse repeated consecutive tokens (HOME HOME HOME -> HOME).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from remote_work_utils import workdays
from remote_work_utils.place2vec.tokenize import tokenize_visits


def build_sentences(
    visits: pd.DataFrame, config: dict[str, Any], calendar: dict[str, Any], scheme: int | None = None
) -> list[list[str]]:
    """``visits`` needs ``identifier``, ``start_date_time``, ``end_date_time``,
    ``duration_hours``, ``date`` plus the tokenisation columns."""
    params = config["corpus"]
    scheme = params["scheme"] if scheme is None else scheme
    visits = visits[visits["duration_hours"] >= params["min_duration_hours"]].copy()
    visits["date"] = pd.to_datetime(visits["date"])
    if params["workdays_only"]:
        visits = visits[~workdays.flag_non_workday(visits["date"], calendar)]
    if "hour" not in visits.columns:
        visits["hour"] = visits["start_date_time"].dt.hour

    poi_counts = visits["poi_id"].dropna().astype(str).value_counts()
    whitelist = set(poi_counts[poi_counts >= params["min_poi_count"]].index)

    visits = visits.sort_values(["identifier", "start_date_time"])
    previous_end = visits.groupby(["identifier", "date"])["end_date_time"].shift(1)
    gap_hours = (visits["start_date_time"] - previous_end).dt.total_seconds() / 3600
    visits["session"] = (gap_hours.isna() | (gap_hours.abs() > params["session_gap_hours"])).cumsum()

    visits["token"] = tokenize_visits(visits, config, scheme, poi_whitelist=whitelist)
    valid = visits[visits["token"] != ""]
    valid = valid[valid["token"] != valid.groupby("session")["token"].shift(1)]
    return [group.tolist() for _, group in valid.groupby("session", sort=False)["token"]]
