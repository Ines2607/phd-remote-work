"""Radius of gyration and local spatial entropy.

Step 1 helpers of ``3302-third-places-macro-micro-ddd``. Radius, metric CRS
and rounding come from the ``mobility`` block of ``configs/third_places.json``
(see :func:`mobility_from_config`).

Required columns (rename via arguments if your schema differs):

- gyration: ``identifier``, ``lon``, ``lat``; optional ``poi_id`` if ``poi_only``
- entropy: ``identifier``, ``dist_to_home`` (metres); plus ``lon``/``lat``
  or ``location_col``. Named POIs are optional (``named_only=True``).

Dependencies: pandas, numpy, geopandas.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import geopandas as gpd

# Default empty-string sentinels used in 3302 for unnamed POIs.
DEFAULT_POI_MISSING: tuple[str, ...] = ("", "None")


def _named_poi_mask(
    poi: pd.Series,
    missing_values: Sequence[str] = DEFAULT_POI_MISSING,
) -> pd.Series:
    """True where poi_id is present and not a missing-token string."""
    s = poi.astype("string")
    missing = {str(v) for v in missing_values}
    return poi.notna() & ~s.isin(missing)


def calculate_radius_of_gyration(
    dwells_df: pd.DataFrame,
    *,
    poi_only: bool = False,
    id_col: str = "identifier",
    lon_col: str = "lon",
    lat_col: str = "lat",
    poi_col: str = "poi_id",
    poi_missing: Sequence[str] = DEFAULT_POI_MISSING,
    crs_geographic: str = "EPSG:4326",
    crs_projected: str | int,
) -> pd.DataFrame:
    """Vectorized radius of gyration (metres) in ``crs_projected``.

    One row per ``id_col``. Formula: RMS Euclidean distance of dwell points
    from the user's centre of mass. ``crs_projected`` must be a metric CRS.

    Parameters
    ----------
    dwells_df
        One row per dwell (or ping).
    poi_only
        If True, keep only named POIs (``poi_col`` not null / missing-token).
    id_col, lon_col, lat_col, poi_col
        Column names.
    poi_missing
        String values treated as unnamed / missing POI ids.
    crs_geographic
        CRS of ``lon`` / ``lat``.
    crs_projected
        Metric CRS used for (x, y) before the RMS.

    Returns
    -------
    DataFrame with columns ``[id_col, "rg"]``. Empty input → empty frame.
    """
    df = dwells_df
    if poi_only:
        df = df.loc[_named_poi_mask(df[poi_col], poi_missing)]
    if df.empty:
        return pd.DataFrame(columns=[id_col, "rg"])

    gdf = gpd.GeoDataFrame(
        df[[id_col]].copy(),
        geometry=gpd.points_from_xy(
            df[lon_col].astype(float), df[lat_col].astype(float)
        ),
        crs=crs_geographic,
    )
    xy = gdf.to_crs(crs_projected)
    coords = pd.DataFrame(
        {
            id_col: xy[id_col].to_numpy(),
            "x": xy.geometry.x,
            "y": xy.geometry.y,
        }
    )
    com = coords.groupby(id_col, sort=False)[["x", "y"]].transform("mean")
    d2 = (coords["x"] - com["x"]) ** 2 + (coords["y"] - com["y"]) ** 2
    rg = np.sqrt(d2.groupby(coords[id_col], sort=False).mean())
    return rg.rename("rg").reset_index()


def _location_keys(
    df: pd.DataFrame,
    *,
    location_col: str | None,
    lon_col: str,
    lat_col: str,
    coord_decimals: int | None,
) -> pd.Series:
    """Categorical location id: ``location_col``, else rounded lon/lat."""
    if location_col is not None:
        return df[location_col]
    lon = df[lon_col].astype(float)
    lat = df[lat_col].astype(float)
    if coord_decimals is not None:
        lon = lon.round(coord_decimals)
        lat = lat.round(coord_decimals)
    return lon.astype(str) + "," + lat.astype(str)


def calculate_local_spatial_entropy(
    dwells_df: pd.DataFrame,
    *,
    radius_m: float,
    id_col: str = "identifier",
    dist_col: str = "dist_to_home",
    location_col: str | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    coord_decimals: int | None = None,
    named_only: bool = False,
    poi_col: str = "poi_id",
    poi_missing: Sequence[str] = DEFAULT_POI_MISSING,
    log=np.log,
) -> pd.DataFrame:
    """Shannon entropy of local visit shares within ``radius_m`` of home.

    Default: **all dwells** inside the radius, locations = lon/lat
    (``coord_decimals``). Pass ``location_col`` (geohash, venue id, …) to
    group by that instead. Named POIs only if ``named_only=True``.

    Users with one local location → 0. Users with none → NaN.
    Units are nats if ``log`` is ``np.log`` (default), bits if ``np.log2``.

    Parameters
    ----------
    dwells_df
        One row per dwell. ``dist_col`` in metres.
    radius_m
        Keep dwells with ``dist_col <= radius_m``.
    location_col
        Location category. ``None`` (default) uses ``lon``/``lat``.
    coord_decimals
        Round lon/lat before grouping when ``location_col`` is None.
        ``None`` keeps full float precision.
    named_only
        If True, keep only named POIs (``poi_col`` not null / missing-token)
        and group by ``poi_col`` unless ``location_col`` is set.
    poi_col, poi_missing
        Used only when ``named_only=True``.
    log
        Logarithm used in ``-sum p log p``.
    """
    ids = dwells_df[[id_col]].drop_duplicates()
    mask = dwells_df[dist_col].le(radius_m)
    if named_only:
        mask = mask & _named_poi_mask(dwells_df[poi_col], poi_missing)
        if location_col is None:
            location_col = poi_col
    local = dwells_df.loc[mask].copy()
    if local.empty:
        return ids.assign(entropy_local=np.nan)

    local["_loc"] = _location_keys(
        local,
        location_col=location_col,
        lon_col=lon_col,
        lat_col=lat_col,
        coord_decimals=coord_decimals,
    )
    local = local.loc[local["_loc"].notna() & (local["_loc"].astype(str) != "nan")]
    if local.empty:
        return ids.assign(entropy_local=np.nan)

    counts = (
        local.groupby([id_col, "_loc"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    tot = counts.groupby(id_col, sort=False)["n"].transform("sum")
    p = counts["n"] / tot
    plogp = p * log(p)
    entropy = -plogp.groupby(counts[id_col], sort=False).sum()
    n_loc = counts.groupby(id_col, sort=False)["_loc"].nunique()
    entropy = entropy.where(n_loc >= 1, np.nan)
    entropy.loc[n_loc == 1] = 0.0
    return ids.merge(
        entropy.rename("entropy_local").reset_index(), on=id_col, how="left"
    )


def user_period_mobility(
    dwells_df: pd.DataFrame,
    *,
    period_col: str = "month",
    radius_m: float,
    id_col: str = "identifier",
    lon_col: str = "lon",
    lat_col: str = "lat",
    poi_col: str = "poi_id",
    dist_col: str = "dist_to_home",
    home_col: str | None = None,
    poi_missing: Sequence[str] = DEFAULT_POI_MISSING,
    crs_geographic: str = "EPSG:4326",
    crs_projected: str | int,
    entropy_named_only: bool = False,
    entropy_location_col: str | None = None,
    coord_decimals: int | None = None,
    log=np.log,
) -> pd.DataFrame:
    """Gyration (all / named POI) and local entropy inside each time period.

    Entropy defaults to **all dwells** within ``radius_m`` (locations = lon/lat).
    Set ``entropy_named_only=True`` for the 3302 named-POI version, or pass
    ``entropy_location_col`` (e.g. geohash) to group by that column.

    Returns one row per ``id_col`` × ``period_col``, with ``rg_all``, ``rg_poi``,
    ``entropy_local``, and ``home_col`` if that column is present.

    ``home_col`` (e.g. the home statistical-area id) is carried through when given.
    """
    rg_kwargs = dict(
        id_col=id_col,
        lon_col=lon_col,
        lat_col=lat_col,
        poi_col=poi_col,
        poi_missing=poi_missing,
        crs_geographic=crs_geographic,
        crs_projected=crs_projected,
    )
    ent_kwargs = dict(
        radius_m=radius_m,
        id_col=id_col,
        dist_col=dist_col,
        location_col=entropy_location_col,
        lon_col=lon_col,
        lat_col=lat_col,
        coord_decimals=coord_decimals,
        named_only=entropy_named_only,
        poi_col=poi_col,
        poi_missing=poi_missing,
        log=log,
    )

    frames: list[pd.DataFrame] = []
    for period, part in dwells_df.groupby(period_col, sort=False):
        rg_all = calculate_radius_of_gyration(
            part, poi_only=False, **rg_kwargs
        ).rename(columns={"rg": "rg_all"})
        rg_poi = calculate_radius_of_gyration(
            part, poi_only=True, **rg_kwargs
        ).rename(columns={"rg": "rg_poi"})
        ent = calculate_local_spatial_entropy(part, **ent_kwargs)
        um = rg_all.merge(rg_poi, on=id_col, how="outer").merge(
            ent, on=id_col, how="outer"
        )
        if home_col is not None and home_col in part.columns:
            home = (
                part.groupby(id_col, sort=False)[home_col]
                .first()
                .reset_index()
            )
            um = um.merge(home, on=id_col, how="left")
        um[period_col] = period
        frames.append(um)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mobility_from_config(dwells_df: pd.DataFrame, params: dict, **kwargs) -> pd.DataFrame:
    """:func:`user_period_mobility` with radius, CRS and rounding from ``params``."""
    return user_period_mobility(
        dwells_df,
        radius_m=params["local_radius_m"],
        crs_projected=params["projected_crs"],
        coord_decimals=params["coord_decimals"],
        **kwargs,
    )
