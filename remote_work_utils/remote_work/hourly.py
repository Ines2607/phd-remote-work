"""Hourly location panel and per-user hour-of-day probabilities.

Port of the first half of ``020-process-dwells/003-remote-work-monthly.py``;
see ``notebook_docs/methodology/REMOTE_WORK_FLAGS.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from remote_work_utils import workdays
from remote_work_utils.geo import distance_m, geohash_polygons
from remote_work_utils.hours import explode_hours

PROBABILITY_COLUMNS = ["prob_home_at_hour", "prob_office_at_hour", "prob_3rdplace_at_hour"]
_LOCATION_NAMES = {"work": "office", "home": "home", "other": "3rdplace"}


def label_locations(dwells: pd.DataFrame, users_work_home: pd.DataFrame, settings: dict[str, Any], month: str) -> pd.DataFrame:
    """Dwells of users with home and work, labelled ``home`` / ``work`` / ``other``.

    A dwell is at home (work) if its centroid is within ``location_radius_meters``
    of the home (work) geohash cell; home wins when both apply.
    """
    radius = settings["remote_work"]["location_radius_meters"]
    swapped = settings["geohash"]["lonlat_swapped"]
    users = users_work_home.drop_duplicates(["identifier"])[["identifier", "geohash_home", "geohash_work"]].assign(
        month=month
    )
    labelled = pd.DataFrame(dwells).assign(month=month).merge(users, on=["identifier", "month"], how="inner")

    labelled["geometry_work"] = geohash_polygons(labelled["geohash_work"], swapped)
    labelled["distance_work"] = distance_m(labelled["geometry_work"], labelled["the_geom"], crs=4326)
    labelled["flag_work_geohash"] = labelled["distance_work"] <= radius
    labelled["geometry_home"] = geohash_polygons(labelled["geohash_home"], swapped)
    labelled["distance_home"] = distance_m(labelled["geometry_home"], labelled["the_geom"], crs=4326)
    labelled["flag_home_geohash"] = labelled["distance_home"] <= radius
    labelled["flag_other_geohash"] = ~labelled["flag_home_geohash"] & ~labelled["flag_work_geohash"]
    labelled["location_type"] = np.select(
        [labelled["flag_home_geohash"], labelled["flag_work_geohash"]], ["home", "work"], default="other"
    )
    return labelled


def build_hours(labelled: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Expand to hours, flag non-workdays and office days, drop duplicate hours.

    An office day is a workday with at least one hour at the work location.
    When two dwells touch the same clock hour, the first one is kept.
    """
    hours = explode_hours(labelled)
    hours["date"] = pd.to_datetime(hours["date"])
    hours["flag_weekend"] = workdays.flag_non_workday(hours["date"], settings["calendar"])
    hours["work_location_hours"] = hours.groupby(["date", "identifier"])["flag_work_geohash"].transform("sum")
    hours["office_day"] = (hours["work_location_hours"] > 0) & (~hours["flag_weekend"])
    return hours.drop_duplicates(["identifier", "hour", "date"])


def _prob_location(location_type: str, hours: pd.DataFrame) -> pd.DataFrame:
    """P(location) on office days: share of a user's office-day hours at the location."""
    return (
        hours[hours["office_day"]]
        .groupby(["identifier", "month"])[f"flag_{location_type}_geohash"]
        .mean()
        .rename(f"prob_{_LOCATION_NAMES[location_type]}")
        .to_frame()
    )


def _prob_hour_given_location(location_type: str, hours: pd.DataFrame) -> pd.DataFrame:
    """P(hour | location) on office days."""
    at_location = hours[(hours["office_day"]) & (hours[f"flag_{location_type}_geohash"])]
    return (
        (at_location.groupby(["identifier", "month", "hour"]).size() / at_location.groupby(["identifier", "month"]).size())
        .rename(f"prob_hour_at_{_LOCATION_NAMES[location_type]}")
        .to_frame()
    )


def conditional_probability_table(hours: pd.DataFrame) -> pd.DataFrame:
    """Bayes: P(location | hour) = P(hour | location) P(location) / P(hour), office days only."""
    office = hours[hours["office_day"]]
    prob_hour = (
        (office.groupby(["identifier", "month", "hour"]).size() / office.groupby(["identifier", "month"]).size())
        .rename("prob_hour")
        .to_frame()
    )
    table = prob_hour
    for location_type in ["work", "home", "other"]:
        table = table.join(_prob_location(location_type, hours), how="left")
    for location_type in ["work", "home", "other"]:
        table = table.join(_prob_hour_given_location(location_type, hours), how="left")
    table = table.fillna(0).reset_index()

    table["prob_office_at_hour"] = table["prob_hour_at_office"] * table["prob_office"] / table["prob_hour"]
    table["prob_home_at_hour"] = table["prob_hour_at_home"] * table["prob_home"] / table["prob_hour"]
    table["prob_3rdplace_at_hour"] = table["prob_hour_at_3rdplace"] * table["prob_3rdplace"] / table["prob_hour"]
    table["flag_weekend"] = False
    return table


def build_hourly_panel(
    dwells: pd.DataFrame, users_work_home: pd.DataFrame, settings: dict[str, Any], month: str
) -> pd.DataFrame:
    """Labelled hourly panel with per-user probabilities (zero on non-workdays)."""
    hours = build_hours(label_locations(dwells, users_work_home, settings, month), settings)
    probabilities = conditional_probability_table(hours)
    return hours.merge(
        probabilities[["identifier", "month", "hour", "flag_weekend", *PROBABILITY_COLUMNS]],
        on=["identifier", "month", "hour", "flag_weekend"],
        how="left",
    ).fillna(0)
