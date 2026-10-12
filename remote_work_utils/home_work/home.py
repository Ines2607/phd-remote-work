"""Monthly home location (night + Shabbat hours).

Port of ``020-process-dwells/001-home-location-monthly.py``; see
``notebook_docs/methodology/HOME_LOCATION.md``.

1. Keep each user's top-N geohashes by number of distinct days.
2. Expand dwells to clock hours.
3. Night candidates: geohashes with at least ``min_night_hours_count`` night hours.
4. Shabbat candidates: hours in the ``shabbat`` window.
5. Weight = night hours + Shabbat hours; the top-weighted geohash is the home
   if it has at least ``min_shabbat_hours_count`` Shabbat hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from remote_work_utils import workdays
from remote_work_utils.geo import geohash_polygons
from remote_work_utils.hours import explode_hours

HOME_COLUMNS = [
    "identifier",
    "geohash",
    "night_hours_count",
    "nights_count",
    "rank_night",
    "shabbat_hours_counts",
    "rank_shabbat",
]


@dataclass
class HomeResult:
    homes: pd.DataFrame
    hours: pd.DataFrame
    dwells_top: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)


def top_geohashes(dwells: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep dwells in each user's ``top_n`` geohashes ranked by distinct days."""
    user_geohash = (
        dwells.groupby(["identifier", "geohash"])["duration_hours"]
        .sum()
        .to_frame()
        .join(dwells.groupby(["identifier", "geohash"])["date"].nunique().to_frame())
    )
    user_geohash["rank_frequency_day"] = user_geohash.groupby(["identifier"])["date"].rank(
        ascending=False, method="first"
    )
    keep = user_geohash[user_geohash["rank_frequency_day"] <= top_n].reset_index()[
        ["identifier", "geohash"]
    ]
    return dwells.merge(keep, how="inner", on=["identifier", "geohash"])


def add_hour_flags(hours: pd.DataFrame, calendar: dict[str, Any]) -> pd.DataFrame:
    """Night / Shabbat / weekend flags used by home detection and diagnostics.

    ``flag_weekend`` here is weekend days only (no holidays), as in the
    original home script.
    """
    hours["flag_shabbat"] = workdays.in_hour_windows(
        hours["date"], hours["hour"], workdays.hour_window(calendar, "shabbat")
    )
    hours["flag_weekend"] = workdays.flag_weekend_only(hours["date"], calendar)
    hours["flag_night"] = workdays.in_hour_windows(
        hours["date"], hours["hour"], workdays.hour_window(calendar, "night")
    )
    hours["flag_work_hour"] = (~hours["flag_weekend"]) & workdays.in_hour_windows(
        hours["date"], hours["hour"], workdays.hour_window(calendar, "core_work")
    )
    hours["flag_home_hour"] = hours["flag_night"] | hours["flag_weekend"]
    return hours


def score_home_candidates(hours: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Night and Shabbat hour counts and ranks per user-geohash."""
    night = hours[hours["flag_night"]].groupby(["identifier", "geohash"])["date"].agg(["count", "nunique"])
    night = night[night["count"] >= params["min_night_hours_count"]].reset_index()
    night["rank_geohash"] = night.groupby(["identifier"])["count"].rank(ascending=False, method="first")

    shabbat = hours[hours["flag_shabbat"]].groupby(["identifier", "geohash"])["hour"].count().to_frame()
    shabbat["rank"] = shabbat.groupby(["identifier"])["hour"].rank(ascending=False, method="first")

    candidates = night.merge(shabbat, on=["identifier", "geohash"], how="outer").fillna(0)
    candidates.columns = HOME_COLUMNS
    candidates["weight"] = candidates["night_hours_count"] + candidates["shabbat_hours_counts"]
    candidates["rank_weighted"] = candidates.groupby("identifier")["weight"].rank(
        ascending=False, method="first"
    )
    return candidates


def select_homes(candidates: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return candidates[
        (candidates["rank_weighted"] == 1)
        & (candidates["shabbat_hours_counts"] >= params["min_shabbat_hours_count"])
    ]


def _frequent_users(hours: pd.DataFrame, diagnostics: dict[str, Any]) -> pd.Series:
    """Users with enough clock hours that repeat on several days (home and work hours)."""

    def frequent_hours(flag: str, name: str) -> pd.DataFrame:
        counts = hours.loc[hours[flag]].groupby(["identifier", "hour"]).size().reset_index(name="count")
        return (
            counts.loc[counts["count"] >= diagnostics["frequent_hour_min_count"]]
            .groupby("identifier")
            .size()
            .reset_index(name=name)
        )

    freq = pd.merge(
        frequent_hours("flag_home_hour", "freq_home_hours_count"),
        frequent_hours("flag_work_hour", "freq_work_hours_count"),
        on="identifier",
        how="inner",
    ).fillna(0)
    return freq[
        (freq["freq_work_hours_count"] >= diagnostics["min_frequent_work_hours"])
        & (freq["freq_home_hours_count"] >= diagnostics["min_frequent_home_hours"])
    ]["identifier"]


def add_home_diagnostics(homes: pd.DataFrame, hours: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Work-hours-at-home shares and frequent-user flag (columns of the home table)."""
    home_hours = homes[["identifier", "geohash"]].merge(
        hours[["identifier", "geohash", "hour", "flag_weekend", "flag_home_hour", "flag_work_hour", "date"]],
        how="inner",
        on=["identifier", "geohash"],
    )
    work_hours = hours[hours["flag_work_hour"]]
    home_work_hours = home_hours[home_hours["flag_work_hour"]]

    user_home_days = (
        hours[hours["flag_home_hour"]].groupby(["identifier"])["date"].nunique().rename("date_home_all").reset_index()
    )
    user_home_work_hours = home_work_hours.groupby(["identifier", "geohash"])["hour"].count().rename("hour_work_home")
    user_home_work_days = (
        home_work_hours.groupby(["identifier", "geohash", "date"])
        .size()
        .reset_index(name="count")
        .groupby(["identifier", "geohash"])["date"]
        .nunique()
        .rename("date_work_home")
        .reset_index()
    )
    user_work_hours = work_hours.groupby(["identifier"]).size().rename("hour_work_all").reset_index()
    user_work_days = work_hours.groupby(["identifier"])["date"].nunique().rename("date_work_all").reset_index()
    user_home_total_hours = (
        home_hours.groupby(["identifier", "geohash"])["hour"].count().rename("hour_home_all").to_frame()
    )

    homes = (
        homes.merge(user_home_work_hours, on=["identifier", "geohash"], how="left")
        .merge(user_home_days, on=["identifier"], how="left")
        .merge(user_home_total_hours, on=["identifier", "geohash"], how="left")
        .merge(user_home_work_days, on=["identifier", "geohash"], how="left")
        .merge(user_work_hours, on=["identifier"], how="left")
        .merge(user_work_days, on=["identifier"], how="left")
    ).fillna(0)

    homes["share_work_in_home_hours"] = homes["hour_work_home"] / homes["hour_home_all"]
    homes["share_home_work_hours"] = np.where(
        homes["hour_work_all"] == 0, np.nan, homes["hour_work_home"] / homes["hour_work_all"]
    )
    homes["share_work_in_home_days"] = homes["date_work_home"] / homes["date_home_all"]
    homes["share_home_work_days"] = np.where(
        homes["date_work_all"] == 0, np.nan, homes["date_work_home"] / homes["date_work_all"]
    )
    homes["share_home_hours_home"] = (homes["night_hours_count"] + homes["shabbat_hours_counts"]) / homes[
        "hour_home_all"
    ]
    share_columns = [
        "share_work_in_home_hours",
        "share_home_work_hours",
        "share_work_in_home_days",
        "share_home_work_days",
        "share_home_hours_home",
    ]
    metrics = homes[share_columns].mean().to_dict()

    frequent = _frequent_users(hours, params["diagnostics"])
    homes["flag_frequent_user"] = homes["identifier"].isin(frequent)
    metrics["freq_users"] = int(len(frequent))
    return homes, metrics


def home_spatial_variance(
    dwells_top: pd.DataFrame, homes: pd.DataFrame, lonlat_swapped: bool, projected_crs: int
) -> pd.Series:
    """Per-user variance (km^2) of distances between visited geohashes and home."""
    with_homes = dwells_top.merge(homes[["identifier", "geohash"]], on=["identifier"], suffixes=("", "_home"))
    visited = gpd.GeoSeries(list(geohash_polygons(with_homes["geohash"], lonlat_swapped)), crs=4326)
    home = gpd.GeoSeries(list(geohash_polygons(with_homes["geohash_home"], lonlat_swapped)), crs=4326)
    visited, home = visited.to_crs(projected_crs), home.to_crs(projected_crs)
    distances = pd.Series(visited.distance(home).to_numpy(), index=with_homes.index)
    return distances.groupby(with_homes["identifier"]).var() / 10**6


def detect_homes(dwells: pd.DataFrame, settings: dict[str, Any], diagnostics: bool = True) -> HomeResult:
    """Run home detection on canonical dwells of one month."""
    params = settings["home_location"]
    dwells_top = top_geohashes(dwells, params["top_geohashes_per_user"])
    dwells_top["date"] = pd.to_datetime(dwells_top["date"])
    hours = add_hour_flags(explode_hours(dwells_top), settings["calendar"])

    homes = select_homes(score_home_candidates(hours, params), params)
    metrics: dict[str, Any] = {"num_homes": int(len(homes))}
    if diagnostics and not homes.empty:
        homes, diagnostic_metrics = add_home_diagnostics(homes, hours, params)
        metrics.update(diagnostic_metrics)
        variance = home_spatial_variance(
            dwells_top, homes, settings["geohash"]["lonlat_swapped"], settings["region"]["projected_crs"]
        )
        metrics["spatial_variance"] = float(variance.median())
    return HomeResult(homes=homes.reset_index(drop=True), hours=hours, dwells_top=dwells_top, metrics=metrics)


def homes_by_stat_area(
    homes: pd.DataFrame, stat_areas: gpd.GeoDataFrame, settings: dict[str, Any]
) -> tuple[gpd.GeoDataFrame, float]:
    """Count homes per statistical area and the Spearman correlation with population.

    A home geohash intersecting several areas is counted in each of them.
    """
    columns = settings["stat_areas"]
    area_id = columns["id_column"]
    per_geohash = homes.groupby(["geohash"])["identifier"].nunique().reset_index()
    per_geohash = gpd.GeoDataFrame(
        per_geohash,
        geometry=geohash_polygons(per_geohash["geohash"], settings["geohash"]["lonlat_swapped"]),
        crs=4326,
    )
    stat_areas = stat_areas.to_crs(4326)
    joined = stat_areas[[area_id, "geometry"]].sjoin(per_geohash, predicate="intersects", how="inner")
    users_per_area = joined.groupby(area_id)["identifier"].sum().reset_index()

    keep = [area_id, columns["name_column"], columns["population_column"], *columns["extra_columns"], "geometry"]
    areas = stat_areas[keep].merge(users_per_area, on=area_id, how="left").fillna(0).reset_index()
    areas = gpd.GeoDataFrame(areas, geometry="geometry", crs=4326)

    for_corr = areas.loc[
        ~areas[columns["name_column"]].isin(columns["exclude_names_from_correlation"]),
        [columns["population_column"], "identifier"],
    ]
    correlation = float(for_corr.corr("spearman").iloc[0, 1])
    return areas, correlation
