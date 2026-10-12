"""Monthly work location (frequent non-home place in working time).

Port of ``020-process-dwells/002-work-location-monthly.py``; see
``notebook_docs/methodology/WORK_LOCATION.md``.

Candidates are user-geohashes visited outside home hours (night, weekends,
holidays, holiday eves) that

* have at least ``min_candidate_hours`` hours on ``min_candidate_days`` days,
* reach ``min_candidate_hours_one_day`` hours on at least one day,
* are more than ``min_distance_home_work_meters`` from home,
* have no Shabbat-window hours and no public-holiday hours.

The candidate with most hours is the work location.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from remote_work_utils import workdays
from remote_work_utils.geo import distance_m, geohash_polygons
from remote_work_utils.hours import explode_hours


def attach_homes(dwells: pd.DataFrame, homes: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Dwells of users with a home, with weekend flag and distance to home."""
    calendar = settings["calendar"]
    swapped = settings["geohash"]["lonlat_swapped"]
    with_home = homes[["identifier", "geohash", "night_hours_count", "nights_count"]].merge(
        pd.DataFrame(dwells), on="identifier", how="inner", suffixes=("_home", "")
    )
    with_home["date"] = pd.to_datetime(with_home["date"])
    with_home["flag_weekend"] = workdays.flag_non_workday(with_home["date"], calendar)
    with_home["home_geometry"] = geohash_polygons(with_home["geohash_home"], swapped)
    with_home["geometry"] = geohash_polygons(with_home["geohash"], swapped)
    with_home["distance_home"] = distance_m(with_home["home_geometry"], with_home["geometry"], crs=4326)
    with_home["flag_home_geohash"] = with_home["distance_home"] <= settings["work_location"]["home_radius_meters"]
    return with_home


def score_work_candidates(
    with_home: pd.DataFrame, hours: pd.DataFrame, homes: pd.DataFrame, settings: dict[str, Any]
) -> pd.DataFrame:
    """Filter and rank work candidates; returns all passing candidates."""
    params = settings["work_location"]
    calendar = settings["calendar"]
    out_of_home = hours[~hours["flag_home_hour"]]
    candidates = (
        out_of_home.groupby(["identifier", "geohash"])
        .size()
        .rename("hours_count")
        .to_frame()
        .join(out_of_home.groupby(["identifier", "geohash"])["date"].nunique().rename("days_count").to_frame())
        .join(
            out_of_home.groupby(["identifier", "geohash", "date"])
            .size()
            .groupby(["identifier", "geohash"])
            .max()
            .rename("hours_max_day_count")
            .to_frame()
        )
        .reset_index()
    )
    candidates = candidates[
        (candidates["hours_count"] >= params["min_candidate_hours"])
        & (candidates["days_count"] >= params["min_candidate_days"])
        & (candidates["hours_max_day_count"] >= params["min_candidate_hours_one_day"])
    ]
    candidates = candidates.merge(
        homes[["identifier", "geohash", "rank_night"]], how="left", on=["identifier", "geohash"]
    ).merge(
        with_home[["identifier", "geohash", "distance_home"]].drop_duplicates(),
        on=["identifier", "geohash"],
    )
    candidates = candidates[candidates["distance_home"] > params["min_distance_home_work_meters"]]

    hours = hours.assign(
        flag_shabbat=workdays.in_hour_windows(hours["date"], hours["hour"], workdays.hour_window(calendar, "shabbat")),
        flag_holiday=workdays.flag_holiday(hours["date"], calendar),
    )
    shabbat_hours = hours.groupby(["identifier", "geohash"])["flag_shabbat"].sum()
    holiday_hours = hours.groupby(["identifier", "geohash"])["flag_holiday"].sum()
    candidates = candidates.merge(
        shabbat_hours[shabbat_hours == 0].reset_index(), on=["identifier", "geohash"], how="inner"
    ).merge(
        holiday_hours[holiday_hours == 0].rename("date").reset_index(), on=["identifier", "geohash"], how="inner"
    )
    candidates["rank_hours_out_home"] = candidates.groupby("identifier")["hours_count"].rank(
        ascending=False, method="first"
    )
    return candidates


def detect_work(dwells: pd.DataFrame, homes: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Return one row per user with both home and work (``users_work_home`` table).

    Columns include ``geohash_home``, ``geohash_work``, ``distance_home`` (home to
    work, m), ``share_days_work_location`` and all columns of ``homes``.
    """
    with_home = attach_homes(dwells, homes, settings)
    hours = explode_hours(with_home)
    hours["flag_night"] = workdays.in_hour_windows(
        hours["date"], hours["hour"], workdays.hour_window(settings["calendar"], "night")
    )
    hours["flag_home_hour"] = hours["flag_night"] | hours["flag_weekend"]

    candidates = score_work_candidates(with_home, hours, homes, settings)
    work = candidates[candidates["rank_hours_out_home"] == 1]

    with_work = with_home.merge(work[["identifier", "geohash"]], how="left", on=["identifier"], suffixes=("", "_work"))
    with_work["flag_work_geohash"] = with_work["geohash"] == with_work["geohash_work"]

    days = (
        with_work.groupby("identifier")["date"]
        .nunique()
        .to_frame()
        .join(
            with_work[with_work["flag_work_geohash"]].groupby("identifier")["date"].nunique().rename("work_date").to_frame(),
            how="inner",
        )
    )
    days["share_days_work_location"] = days["work_date"] / days["date"]
    days = days.merge(
        with_work[with_work["flag_work_geohash"]][["identifier", "distance_home", "home_geometry"]].drop_duplicates(),
        on="identifier",
        how="left",
    )
    dashboard = days.merge(homes, on="identifier", how="inner").merge(
        work[["identifier", "geohash"]], on=["identifier"], suffixes=("_home", "_work"), how="inner"
    )
    if dashboard["identifier"].nunique() != len(dashboard):
        raise AssertionError("More than one work location per user")
    return dashboard
