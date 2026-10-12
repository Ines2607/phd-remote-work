"""External validation of detected homes, workplaces and the remote-work trend.

Generic versions of ``XXX-calculate-WFH/203`` (homes vs census population),
``204`` (workplaces vs employment zones) and ``200`` (WFH vs public-transport
ticket validations).
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import shapely
import numpy as np
import pandas as pd

from remote_work_utils.geo import geohash_polygons


def homes_vs_population(
    homes_per_area: pd.DataFrame, population: pd.DataFrame, area_id: str, population_column: str
) -> pd.DataFrame:
    """Per month: Pearson / Spearman correlation of home counts with population and coverage.

    ``homes_per_area`` has ``[area_id, month, num_homes]``; areas without homes count as 0.
    """
    months = homes_per_area[["month"]].drop_duplicates()
    grid = population[[area_id, population_column]].drop_duplicates().merge(months, how="cross")
    full = grid.merge(homes_per_area, on=[area_id, "month"], how="left").fillna({"num_homes": 0})
    rows = []
    for month, group in full.groupby("month"):
        slope, intercept = np.polyfit(group["num_homes"], group[population_column], 1)
        rows.append(
            {
                "month": month,
                "pearson": group[[population_column, "num_homes"]].corr().iloc[0, 1],
                "spearman": group[[population_column, "num_homes"]].corr("spearman").iloc[0, 1],
                "share_areas_with_homes": (group["num_homes"] > 0).mean(),
                "residents_per_detected_home": slope,
                "intercept": intercept,
            }
        )
    return pd.DataFrame(rows)


def workplaces_vs_employment(
    work_counts: pd.DataFrame,
    zones: gpd.GeoDataFrame,
    zone_id: str,
    employment_column: str,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    """Detected workplaces per employment zone (by geohash centroid) and their correlation with jobs.

    ``work_counts`` has ``[geohash_work, num_works]``.
    """
    cells = gpd.GeoDataFrame(
        work_counts,
        geometry=list(geohash_polygons(work_counts["geohash_work"], settings["geohash"]["lonlat_swapped"])),
        crs=4326,
    )
    cells = cells.set_geometry(gpd.GeoSeries(shapely.centroid(cells.geometry.values), crs=4326, index=cells.index))
    joined = gpd.sjoin(cells, zones.to_crs(4326)[[zone_id, employment_column, "geometry"]], predicate="within")
    per_zone = joined.groupby(zone_id).agg(employment=(employment_column, "mean"), num_works=("num_works", "sum"))
    return per_zone.reset_index(), float(per_zone.corr().iloc[0, 1])


def wfh_vs_series(trend: pd.DataFrame, series: pd.DataFrame, value_column: str, metric: str = "wfh_mean") -> float:
    """Correlation between ``1 - WFH`` and an external monthly series (e.g. ticket validations).

    Both frames need a ``month`` column (``YYYYMM``).
    """
    merged = trend.assign(month=trend["month"].astype(str)).merge(
        series.assign(month=series["month"].astype(str)), on="month"
    )
    return float((1 - merged[metric]).corr(merged[value_column]))
