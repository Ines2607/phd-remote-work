"""From user-month remote-work values to monthly trends and area panels.

Follows ``XXX-calculate-WFH/100`` / ``101`` (monthly and statistical-area
means) and ``900-features-stat-areas/110`` (quarterly best month).
"""

from __future__ import annotations

from typing import Any, Iterable

import geopandas as gpd
import shapely
import pandas as pd

from remote_work_utils.geo import geohash_polygons
from remote_work_utils.remote_work.metrics import SELECTED_METRIC


def load_monthly_outputs(settings: dict[str, Any], name: str, months: Iterable[str]) -> pd.DataFrame:
    """Concatenate a per-month output (e.g. ``remote_work_users``) with a ``month`` column."""
    from remote_work_utils.pipeline import output_path

    frames = []
    for month in months:
        path = output_path(settings, name, month)
        if path.exists():
            frames.append(pd.read_csv(path).assign(month=str(month)))
    if not frames:
        raise FileNotFoundError(f"No '{name}' outputs found for months {list(months)}")
    return pd.concat(frames, ignore_index=True)


def filter_user_months(user_months: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Keep user-months with enough detected working days (and selected device types)."""
    params = settings["aggregation"]
    kept = user_months[user_months["total_work_days"] >= params["min_total_work_days"]]
    if params["identifier_types"]:
        kept = kept[kept["identifier_type"].isin(params["identifier_types"])]
    return kept.copy()


def monthly_trend(user_months: pd.DataFrame, metric: str = SELECTED_METRIC, by: list[str] | None = None) -> pd.DataFrame:
    """Mean metric and number of users per month (optionally split by ``by``)."""
    keys = ["month", *(by or [])]
    return (
        user_months.groupby(keys)[metric]
        .agg(wfh_mean="mean", num_users="count")
        .reset_index()
    )


def geohash_area_lookup(geohashes: Iterable[str], stat_areas: gpd.GeoDataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Assign each geohash to the statistical area containing its centroid."""
    area_id = settings["stat_areas"]["id_column"]
    unique = pd.Series(pd.unique(pd.Series(list(geohashes)).dropna()), name="geohash")
    cells = gpd.GeoDataFrame(
        {"geohash": unique},
        geometry=list(geohash_polygons(unique, settings["geohash"]["lonlat_swapped"])),
        crs=4326,
    )
    centroids = cells.set_geometry(gpd.GeoSeries(shapely.centroid(cells.geometry.values), crs=4326, index=cells.index))
    joined = gpd.sjoin(centroids, stat_areas.to_crs(4326)[[area_id, "geometry"]], how="inner", predicate="within")
    return pd.DataFrame(joined[["geohash", area_id]])


def stat_area_monthly(
    user_months: pd.DataFrame,
    homes_months: pd.DataFrame,
    lookup: pd.DataFrame,
    settings: dict[str, Any],
    metric: str = SELECTED_METRIC,
) -> pd.DataFrame:
    """Mean metric of residents (by home geohash) per statistical area and month.

    ``num_homes`` is the number of users behind each mean; rows with fewer than
    ``aggregation.min_homes_per_area`` should not be published.
    """
    area_id = settings["stat_areas"]["id_column"]
    merged = (
        user_months[["identifier", metric, "month"]]
        .merge(homes_months[["identifier", "geohash", "month"]], on=["identifier", "month"], how="inner")
        .merge(lookup, on="geohash", how="inner")
    )
    return merged.groupby([area_id, "month"])[metric].agg(num_homes="count", WFH_mean="mean").reset_index()


def quarterly_best_month(area_months: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """One row per area and quarter: the month with enough homes (most homes on ties).

    The quarterly value is the selected month's value, not a three-month average.
    """
    area_id = settings["stat_areas"]["id_column"]
    minimum = settings["aggregation"]["min_homes_per_area"]
    ranked = area_months.assign(
        quarter=pd.PeriodIndex(pd.to_datetime(area_months["month"].astype(str), format="%Y%m"), freq="Q").astype(str),
        enough_homes=area_months["num_homes"] >= minimum,
    ).sort_values([area_id, "quarter", "enough_homes", "num_homes", "month"], ascending=[True, True, False, False, True])
    best = ranked.drop_duplicates([area_id, "quarter"]).rename(columns={"month": "selected_month"})
    return best.drop(columns="enough_homes").reset_index(drop=True)
