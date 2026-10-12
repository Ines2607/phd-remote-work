"""Turn visits into place2vec tokens.

Input columns (one row per visit): ``flag_home``, ``flag_work``,
``dwell_geohash``, ``poi_id``, ``pois_category``, ``context_type``,
``census_id_filled`` (area of the visit), ``hour``, ``home_census``,
``work_census`` and, for scheme 5 only, ``poi_location_suits_rw``.
``context_type`` (land-use context of cells without a POI) may be absent.

Schemes (from the paper):

====  ==========================  =============================  ======================
id    home / work                 visit with a POI               visit without a POI
====  ==========================  =============================  ======================
0     HOME / WORK                 CAT_<category>                 CTX_<context>
1     HOME_<area> / WORK_<area>   POI_<id>                       GH_<geohash8>
2     HOME_<area> / WORK_<area>   POI_<id>                       CENSUS_<area> or GH_<gh6>
3     HOME_<area> / WORK_<area>   POI_<id>_WEEKDAY_<part>        GH_<geohash8>_<part>
5     HOME_<core>_WEEKDAY_<part>  POI_<cat>_<core>_<loc>_...     GH_<core>_WEEKDAY_<part>
====  ==========================  =============================  ======================
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_MISSING = {"nan": "", "None": "", "<NA>": ""}


def day_part(hours: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Day-part label for each hour (``day_parts`` windows, else ``other_day_part``)."""
    labels = pd.Series(config["other_day_part"], index=hours.index, dtype="object")
    for part in config["day_parts"]:
        labels[(hours >= part["start_hour"]) & (hours < part["end_hour"])] = part["name"]
    return labels


def _area_text(area_ids: pd.Series) -> pd.Series:
    """Area ids as text without a float suffix (``1023.0`` -> ``1023``)."""
    return area_ids.astype(object).where(area_ids.notna(), "").astype(str).str.replace(r"\.0$", "", regex=True)


def _core_label(area_ids: pd.Series, scheme5: dict[str, Any]) -> pd.Series:
    ids = _area_text(area_ids)
    return pd.Series(
        np.where(ids.str.startswith(scheme5["core_area_prefix"]), scheme5["core_label"], scheme5["other_label"]),
        index=area_ids.index,
        dtype="object",
    )


def tokenize_visits(visits: pd.DataFrame, config: dict[str, Any], scheme: int, poi_whitelist: set | None = None) -> pd.Series:
    """One token per visit ("" when nothing is produced)."""
    tokens = pd.Series("", index=visits.index, dtype="object")
    flag_home = visits["flag_home"].fillna(False).astype(bool)
    flag_work = visits["flag_work"].fillna(False).astype(bool)
    geohash = visits["dwell_geohash"].fillna("").astype(str)
    poi_id = visits["poi_id"].astype(str).replace(_MISSING)
    has_poi = (poi_id != "") & ~flag_home & ~flag_work
    if poi_whitelist is not None:
        has_poi &= poi_id.isin(poi_whitelist)
    no_poi = ~flag_home & ~flag_work & ~has_poi

    category = visits["pois_category"].fillna("UNKNOWN").astype(str).str.replace(" ", "_")
    raw_context = visits["context_type"] if "context_type" in visits else pd.Series(pd.NA, index=visits.index)
    context = raw_context.astype(str).replace({k: "UNKNOWN" for k in _MISSING}).str.replace(" ", "_")
    area = _area_text(visits["census_id_filled"])
    period = day_part(visits["hour"].fillna(0).astype(int), config)
    home_area = _area_text(visits["home_census"])
    work_area = _area_text(visits["work_census"])
    geohash, poi_id, category, context, area, home_area, work_area = (
        s.astype(object) for s in (geohash, poi_id, category, context, area, home_area, work_area)
    )

    if scheme == 0:
        tokens[flag_home] = "HOME"
        tokens[flag_work] = "WORK"
        tokens[has_poi] = "CAT_" + category[has_poi]
        tokens[no_poi] = "CTX_" + context[no_poi]
        return tokens

    if scheme == 5:
        scheme5 = config["scheme5"]
        tokens[flag_home] = ("HOME_" + _core_label(home_area, scheme5) + "_WEEKDAY_" + period)[flag_home]
        tokens[flag_work] = ("WORK_" + _core_label(work_area, scheme5))[flag_work]
        location = pd.Series(
            np.where(visits["poi_location_suits_rw"].fillna(1).astype(float) == 0, "office_loc", "third_place"),
            index=visits.index,
        )
        tokens[has_poi] = (
            "POI_" + category + "_" + _core_label(area, scheme5) + "_" + location + "_WEEKDAY_" + period
        )[has_poi]
        tokens[no_poi] = ("GH_" + _core_label(area, scheme5) + "_WEEKDAY_" + period)[no_poi]
        return tokens

    for flag, area_ids, name in [(flag_home, home_area, "HOME"), (flag_work, work_area, "WORK")]:
        with_area = flag & (area_ids != "")
        tokens[with_area] = name + "_" + area_ids[with_area]
        tokens[flag & (area_ids == "")] = name

    if scheme in (1, 2):
        tokens[has_poi] = "POI_" + poi_id[has_poi]
    elif scheme == 3:
        tokens[has_poi] = "POI_" + poi_id[has_poi] + "_WEEKDAY_" + period[has_poi]
    else:
        raise ValueError(f"Unknown tokenisation scheme {scheme}")

    if scheme == 1:
        tokens[no_poi] = "GH_" + geohash[no_poi].str[:8]
    elif scheme == 2:
        with_area = no_poi & (area != "")
        tokens[with_area] = "CENSUS_" + area[with_area]
        tokens[no_poi & (area == "")] = "GH_" + geohash[no_poi & (area == "")].str[:6]
    else:
        tokens[no_poi] = "GH_" + geohash[no_poi].str[:8] + "_" + period[no_poi]
    return tokens
