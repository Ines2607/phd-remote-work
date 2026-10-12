"""Geohash geometry and distances.

``geohash.lonlat_swapped`` exists for compatibility with the Gush Dan dataset,
whose geohash strings were produced with ``pygeohash.encode(lon, lat)``. Decoding
must then swap the coordinates back. New datasets should use standard geohashes
(``lonlat_swapped: false``).
"""

from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pygeohash
from shapely import wkb, wkt
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry


def encode_geohash(
    latitudes: Iterable[float],
    longitudes: Iterable[float],
    precision: int,
    lonlat_swapped: bool,
) -> list[str]:
    if lonlat_swapped:
        return [pygeohash.encode(lon, lat, precision=precision) for lat, lon in zip(latitudes, longitudes)]
    return [pygeohash.encode(lat, lon, precision=precision) for lat, lon in zip(latitudes, longitudes)]


def geohash_to_polygon(geohash: str, lonlat_swapped: bool) -> Polygon:
    """Bounding box of a geohash cell in EPSG:4326 (x = longitude)."""
    first, second, first_err, second_err = pygeohash.decode_exactly(geohash)
    if lonlat_swapped:
        lon, lat, lon_err, lat_err = first, second, first_err, second_err
    else:
        lat, lon, lat_err, lon_err = first, second, first_err, second_err
    lat_min, lat_max = lat - lat_err, lat + lat_err
    lon_min, lon_max = lon - lon_err, lon + lon_err
    return Polygon(
        [(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min), (lon_min, lat_min)]
    )


def geohash_polygons(geohashes: pd.Series, lonlat_swapped: bool) -> pd.Series:
    """Vectorised :func:`geohash_to_polygon` (each distinct geohash decoded once)."""
    lookup = {gh: geohash_to_polygon(gh, lonlat_swapped) for gh in pd.unique(geohashes.dropna())}
    return geohashes.map(lookup)


def to_geometry(values: pd.Series) -> pd.Series:
    """Parse WKB bytes / WKT strings into shapely geometries (pass through otherwise)."""
    sample = values.dropna()
    if sample.empty or isinstance(sample.iloc[0], BaseGeometry):
        return values
    if isinstance(sample.iloc[0], (bytes, bytearray)):
        return values.apply(lambda x: None if x is None else wkb.loads(x))
    if isinstance(sample.iloc[0], str):
        return values.apply(lambda x: None if pd.isnull(x) else wkt.loads(x))
    raise TypeError(f"Unrecognised geometry type: {type(sample.iloc[0])}")


def points_from_lonlat(longitudes: Iterable[float], latitudes: Iterable[float]) -> list[Point]:
    return [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)]


def distance_m(geometries_a: pd.Series, geometries_b: pd.Series, crs: int | str) -> pd.Series:
    """Distance in metres between two geometry columns.

    Both columns are projected to the UTM zone estimated from ``geometries_a``
    (same behaviour as the original pipeline).
    """
    series_a = gpd.GeoSeries(np.asarray(geometries_a, dtype=object), index=geometries_a.index, crs=crs)
    series_b = gpd.GeoSeries(np.asarray(geometries_b, dtype=object), index=geometries_b.index, crs=crs)
    utm = series_a.estimate_utm_crs()
    return series_a.to_crs(utm).distance(series_b.to_crs(utm))


def spatial_variance_km2(distances_m: pd.Series) -> float:
    """Variance of distances (m) expressed in km^2."""
    return float(np.var(distances_m, ddof=1) / 10**6) if len(distances_m) > 1 else float("nan")
