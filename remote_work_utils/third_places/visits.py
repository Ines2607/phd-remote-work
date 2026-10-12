"""Link dwells to POIs, users' home/work and work schedules."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from remote_work_utils.geo import geohash_polygons


def match_pois(dwells: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, radius_m: float, projected_crs: int) -> pd.Series:
    """Nearest POI id within ``radius_m`` of each dwell centroid (missing otherwise)."""
    left = dwells[[dwells.geometry.name]].to_crs(projected_crs)
    right = pois[["poi_id", pois.geometry.name]].to_crs(projected_crs)
    joined = gpd.sjoin_nearest(left, right, how="left", max_distance=radius_m, distance_col="_distance")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["poi_id"].reindex(dwells.index)


def add_areas(visits: pd.DataFrame, stat_areas: gpd.GeoDataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Add ``census_id_filled`` (area of the dwell cell), ``home_census`` and ``work_census``."""
    from remote_work_utils.remote_work.aggregate import geohash_area_lookup

    area_id = settings["stat_areas"]["id_column"]
    cells = pd.concat([visits["geohash"], visits["geohash_home"], visits["geohash_work"]])
    lookup = geohash_area_lookup(cells, stat_areas, settings).set_index("geohash")[area_id]
    out = visits.copy()
    out["census_id_filled"] = out["geohash"].map(lookup)
    out["home_census"] = out["geohash_home"].map(lookup)
    out["work_census"] = out["geohash_work"].map(lookup)
    return out


def work_schedule(user_months: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """``remote_worker`` / ``neither_onsite_not_remote`` / ``onsite_worker`` per user-month.

    ``user_months`` is the user remote-work table (``user_remote_work_<month>.csv``).
    """
    undefined = (
        user_months["total_work_days"]
        - user_months["office_day_counts"].fillna(0)
        - user_months["remote_work_days_min_work"]
    )
    return pd.Series(
        np.select(
            [
                user_months[params["metric"]] >= params["remote_worker_min_share"],
                undefined / user_months["total_work_days"] > params["max_undefined_share"],
            ],
            ["remote_worker", "neither_onsite_not_remote"],
            default="onsite_worker",
        ),
        index=user_months.index,
    )


def add_visit_context(
    dwells: pd.DataFrame, users_work_home: pd.DataFrame, settings: dict[str, Any], params: dict[str, Any]
) -> pd.DataFrame:
    """Distances to home / work and what the user did just before each dwell.

    Adds ``flag_home``, ``flag_work`` (dwell within ``location_radius_meters``
    of the home / work cell), ``dist_home``, ``dist_work``, ``prev_flag_home``,
    ``prev_flag_work``, ``time_diff_from_prev`` (hours) and ``distance_from_prev`` (m).
    """
    swapped = settings["geohash"]["lonlat_swapped"]
    crs = settings["region"]["projected_crs"]
    radius = settings["remote_work"]["location_radius_meters"]
    users = users_work_home.drop_duplicates("identifier")[["identifier", "geohash_home", "geohash_work"]]
    visits = pd.DataFrame(dwells).merge(users, on="identifier", how="inner")

    points = gpd.GeoSeries(list(visits["the_geom"]), crs=4326, index=visits.index).to_crs(crs)
    for kind in ["home", "work"]:
        cells = gpd.GeoSeries(list(geohash_polygons(visits[f"geohash_{kind}"], swapped)), crs=4326, index=visits.index)
        visits[f"dist_{kind}"] = points.distance(cells.to_crs(crs))
        visits[f"flag_{kind}"] = visits[f"dist_{kind}"] <= radius
    visits.loc[visits["flag_home"], "flag_work"] = False

    visits = visits.sort_values(["identifier", "start_date_time"])
    by_user = visits.groupby("identifier")
    visits["prev_flag_home"] = by_user["flag_home"].shift(1).fillna(False).astype(bool)
    visits["prev_flag_work"] = by_user["flag_work"].shift(1).fillna(False).astype(bool)
    visits["time_diff_from_prev"] = (
        visits["start_date_time"] - by_user["end_date_time"].shift(1)
    ).dt.total_seconds() / 3600
    previous = gpd.GeoSeries(list(by_user["the_geom"].shift(1)), crs=4326, index=visits.index).to_crs(crs)
    visits["distance_from_prev"] = gpd.GeoSeries(points.reindex(visits.index)).distance(previous)
    recent = visits["time_diff_from_prev"] <= params["max_hours_after_previous"]
    visits["after_home"] = visits["prev_flag_home"] & recent
    visits["after_work"] = visits["prev_flag_work"] & recent
    return visits
