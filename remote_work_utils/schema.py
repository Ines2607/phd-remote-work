"""Canonical dwell schema and normalisation of input tables.

Every algorithm in the package works on these columns:

==================  ===========================================================
identifier          device / user id
identifier_type     device type (e.g. GAID = Android, IDFA = iOS)
geohash             geohash cell of the dwell centroid
start_date_time     local start time (tz-naive)
end_date_time       local end time (tz-naive)
date                local date of the dwell start
duration_hours      dwell duration in hours
the_geom            dwell centroid (shapely Point, EPSG:4326)
==================  ===========================================================

``input.columns`` in the settings maps each canonical name to the source column.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from remote_work_utils.geo import to_geometry

CANONICAL_COLUMNS = [
    "identifier",
    "identifier_type",
    "geohash",
    "start_date_time",
    "end_date_time",
    "date",
    "duration_hours",
    "the_geom",
]


def normalize_dwells(df: pd.DataFrame, settings: dict[str, Any]) -> gpd.GeoDataFrame:
    """Rename source columns to the canonical schema and parse types.

    Columns that are not part of the mapping are kept unchanged.
    """
    mapping = settings["input"]["columns"]
    missing = [source for source in mapping.values() if source not in df.columns]
    if missing:
        raise KeyError(f"Input is missing columns {missing} (see input.columns in settings)")
    renamed = df.rename(columns={source: canonical for canonical, source in mapping.items()})
    renamed = renamed.copy()
    renamed["start_date_time"] = pd.to_datetime(renamed["start_date_time"])
    renamed["end_date_time"] = pd.to_datetime(renamed["end_date_time"])
    renamed["the_geom"] = to_geometry(renamed["the_geom"])
    return gpd.GeoDataFrame(renamed, geometry="the_geom", crs=settings["input"]["crs"])


def read_dwells(path, settings: dict[str, Any]) -> gpd.GeoDataFrame:
    """Read a filtered dwell table (Parquet or CSV) into the canonical schema."""
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No dwells in {path}")
    return normalize_dwells(df, settings)
