"""Raw dwells -> filtered monthly dwells in the canonical schema.

Port of ``004-load-dwells/001-create-quarter-filtered-parquet.py`` and
``004-load-dwells/run-monthly-data.py`` (pandas instead of dask).

Raw input columns: ``identifier, identifier_type, timestamp, local_date_time,
province, classification, duration_seconds, centroid_latitude,
centroid_longitude, bump_count``.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkb

from remote_work_utils.geo import encode_geohash, points_from_lonlat


def add_dwell_flags(dwells: pd.DataFrame, flags: dict[str, Any]) -> pd.DataFrame:
    """Dwell-level night / work-time flags used for the user filter."""
    start, end = dwells["start_date_time"], dwells["end_date_time"]
    night_start, night_end = flags["night_start_hour"], flags["night_end_hour"]
    dwells["flag_night"] = (
        (start.dt.hour >= night_start)
        | (start.dt.hour < night_end)
        | (end.dt.hour >= night_start)
        | (end.dt.hour < night_end)
        | ((start.dt.hour < night_start) & (end.dt.hour > night_end) & (start.dt.date != end.dt.date))
    )
    dwells["flag_work_hours"] = (
        start.dt.dayofweek.isin(flags["work_weekdays"])
        & (start.dt.hour < flags["work_end_hour"])
        & (end.dt.hour >= flags["work_start_hour"])
    )
    return dwells


def _users_seen_on_enough_dates(dwells: pd.DataFrame, flag: str, min_dates: int) -> set:
    per_geohash = dwells[dwells[flag]].groupby(["identifier", "geohash"])["date"].nunique()
    best = per_geohash.groupby("identifier").max()
    return set(best[best >= min_dates].index)


def prepare_dwells(
    raw: pd.DataFrame, settings: dict[str, Any], boundary: gpd.GeoDataFrame | None = None
) -> pd.DataFrame:
    """Filter raw dwells and add the canonical columns.

    Keeps dwells longer than ``min_duration_seconds`` (in ``provinces`` and inside
    ``boundary`` if given), users with at least ``min_dwells_per_user`` dwells,
    the configured ``classifications``, and users seen at the same geohash on
    ``min_night_dates`` nights and ``min_work_dates`` workdays.
    ``the_geom`` is stored as WKB so the result can be written to Parquet.
    """
    params = settings["prepare"]
    dwells = raw[raw["duration_seconds"] > params["min_duration_seconds"]].copy()
    if params["provinces"]:
        dwells = dwells[dwells["province"].isin(params["provinces"])]
    dwells["month"] = pd.to_datetime(dwells["timestamp"]).dt.month

    dwells = gpd.GeoDataFrame(
        dwells,
        geometry=points_from_lonlat(dwells["centroid_longitude"], dwells["centroid_latitude"]),
        crs=4326,
    ).rename_geometry("the_geom")
    if params["clip_to_boundary"]:
        if boundary is None:
            raise ValueError("prepare.clip_to_boundary is true but no boundary was given")
        dwells = dwells[dwells.within(boundary.to_crs(4326).union_all())]

    counts = dwells.groupby("identifier").size().reset_index(name="frequency")
    dwells = dwells.merge(counts, on="identifier")
    dwells = dwells[dwells["frequency"] >= params["min_dwells_per_user"]]

    dwells["duration_hours"] = dwells["duration_seconds"] / 3600
    dwells = dwells[dwells["classification"].isin(params["classifications"])].copy()
    dwells["geohash"] = encode_geohash(
        dwells["centroid_latitude"],
        dwells["centroid_longitude"],
        settings["geohash"]["precision"],
        settings["geohash"]["lonlat_swapped"],
    )
    dwells["start_date_time"] = pd.to_datetime(dwells["local_date_time"])
    dwells["date"] = dwells["start_date_time"].dt.date
    dwells["end_date_time"] = dwells["start_date_time"] + pd.to_timedelta(dwells["duration_seconds"], unit="s")
    dwells = add_dwell_flags(dwells, params["dwell_flags"])

    good_ids = _users_seen_on_enough_dates(dwells, "flag_night", params["min_night_dates"]) & (
        _users_seen_on_enough_dates(dwells, "flag_work_hours", params["min_work_dates"])
    )
    good = pd.DataFrame(dwells[dwells["identifier"].isin(good_ids)])
    good["the_geom"] = good["the_geom"].apply(wkb.dumps)
    return good.reset_index(drop=True)
