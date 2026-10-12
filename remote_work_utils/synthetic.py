"""Synthetic GPS dwells with known home, office and third places.

Used by the demo notebooks and tests. Nothing here comes from real devices:
users, places and visits are drawn from the ``synthetic`` settings section.

Behaviour per user and day (local time, workday calendar from settings):

* nights and Shabbat / holidays at home;
* office workers go to the office on workdays, or stay home on remote days
  (probability ``remote_day_probability[month] * personal affinity``);
* on remote days (and every workday of users without an office) some users
  work from a work-friendly third place (cafe, coworking, library) in the
  late morning; evenings and weekend outings go to leisure places;
* each dwell is observed with a user-specific probability (data sparsity).

Day timings are the ``schedule`` templates: ``[kind, [start_min, start_max],
[end_min, end_max]]`` in hours from local midnight, where ``kind`` is
``home``, ``office`` or a key of ``place_categories``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from remote_work_utils import workdays

METERS_PER_DEGREE_LAT = 111_320


@dataclass
class SyntheticWorld:
    users: pd.DataFrame
    places: pd.DataFrame
    stat_areas: gpd.GeoDataFrame
    boundary: gpd.GeoDataFrame


def _meters_to_degrees(meters: np.ndarray, latitude: float) -> tuple[np.ndarray, np.ndarray]:
    dlat = meters / METERS_PER_DEGREE_LAT
    dlon = meters / (METERS_PER_DEGREE_LAT * np.cos(np.radians(latitude)))
    return dlat, dlon


def _distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    dy = (np.asarray(lat2) - lat1) * METERS_PER_DEGREE_LAT
    dx = (np.asarray(lon2) - lon1) * METERS_PER_DEGREE_LAT * np.cos(np.radians(lat1))
    return np.hypot(dx, dy)


def make_stat_areas(params: dict[str, Any], rng: np.random.Generator) -> gpd.GeoDataFrame:
    bbox = params["bbox"]
    rows, cols = params["stat_area_grid"]["rows"], params["stat_area_grid"]["cols"]
    lat_edges = np.linspace(bbox["lat_min"], bbox["lat_max"], rows + 1)
    lon_edges = np.linspace(bbox["lon_min"], bbox["lon_max"], cols + 1)
    low, high = params["population_per_area"]
    records = []
    for i in range(rows):
        for j in range(cols):
            records.append(
                {
                    "stat_area_id": 1000 + i * cols + j,
                    "city_name": f"Synthetic City {'ABC'[min(j * 3 // cols, 2)]}",
                    "population": int(rng.integers(low, high)),
                    "geometry": box(lon_edges[j], lat_edges[i], lon_edges[j + 1], lat_edges[i + 1]),
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=4326)


def make_world(params: dict[str, Any]) -> SyntheticWorld:
    """Draw users, homes, offices and third places."""
    rng = np.random.default_rng(params["seed"])
    stat_areas = make_stat_areas(params, rng)
    bbox = params["bbox"]
    mid_lat = (bbox["lat_min"] + bbox["lat_max"]) / 2
    n_users = params["n_users"]

    weights = stat_areas["population"] / stat_areas["population"].sum()
    home_area = rng.choice(len(stat_areas), size=n_users, p=weights)
    bounds = stat_areas.geometry.bounds.to_numpy()[home_area]
    home_lon = rng.uniform(bounds[:, 0], bounds[:, 2])
    home_lat = rng.uniform(bounds[:, 1], bounds[:, 3])

    hubs = params["office_hubs"]
    n_hubs = max(hubs["min_hubs"], n_users // hubs["users_per_hub"])
    hub_lat = rng.uniform(bbox["lat_min"], bbox["lat_max"], n_hubs)
    hub_lon = rng.uniform(bbox["lon_min"], bbox["lon_max"], n_hubs)
    has_office = rng.random(n_users) < params["share_with_office"]
    office_lat = np.full(n_users, np.nan)
    office_lon = np.full(n_users, np.nan)
    for user in np.flatnonzero(has_office):
        far = np.flatnonzero(
            _distance_m(home_lat[user], home_lon[user], hub_lat, hub_lon) >= params["min_home_office_distance_m"]
        )
        if far.size == 0:
            has_office[user] = False
            continue
        hub = rng.choice(far)
        dlat, dlon = _meters_to_degrees(rng.normal(0, hubs["office_jitter_m"], 2), mid_lat)
        office_lat[user], office_lon[user] = hub_lat[hub] + dlat[0], hub_lon[hub] + dlon[1]

    types = list(params["identifier_type_shares"])
    shares = np.array([params["identifier_type_shares"][t] for t in types])
    low_aff, high_aff = params["office_worker_remote_affinity"]
    users = pd.DataFrame(
        {
            "identifier": [f"synthetic-{i:05d}" for i in range(n_users)],
            "identifier_type": rng.choice(types, size=n_users, p=shares / shares.sum()),
            "home_stat_area_id": stat_areas["stat_area_id"].to_numpy()[home_area],
            "home_lat": home_lat,
            "home_lon": home_lon,
            "has_office": has_office,
            "office_lat": office_lat,
            "office_lon": office_lon,
            "remote_affinity": rng.uniform(low_aff, high_aff, n_users),
            "observation_rate": np.clip(
                rng.normal(params["observation_probability"], params["observation_sd"], n_users),
                *params["observation_bounds"],
            ),
        }
    )

    n_places = params["n_third_places"]
    groups = params["place_categories"]
    group = rng.choice(list(groups), size=n_places, p=[params["place_group_shares"][g] for g in groups])
    places = pd.DataFrame(
        {
            "place_id": [f"place-{i:03d}" for i in range(n_places)],
            "category": [rng.choice(groups[g]) for g in group],
            "place_group": group,
            "lat": rng.uniform(bbox["lat_min"], bbox["lat_max"], n_places),
            "lon": rng.uniform(bbox["lon_min"], bbox["lon_max"], n_places),
        }
    )
    favourites = params["favourite_places"]
    for name in groups:
        candidates = places[places["place_group"] == name]
        chosen = []
        for user in users.itertuples():
            distance = _distance_m(user.home_lat, user.home_lon, candidates["lat"], candidates["lon"])
            nearest = np.argsort(distance)[: favourites["nearest_candidates"]]
            size = min(favourites["per_group"], len(nearest))
            chosen.append(list(candidates["place_id"].to_numpy()[rng.choice(nearest, size=size, replace=False)]))
        users[f"favourite_{name}"] = chosen

    boundary = gpd.GeoDataFrame(
        {"name": ["synthetic boundary"]},
        geometry=[box(bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]).buffer(0.01)],
        crs=4326,
    )
    return SyntheticWorld(users=users, places=places, stat_areas=stat_areas, boundary=boundary)


def _day_template(user, day: pd.Timestamp, non_workday: bool, remote_p: float, params: dict, rng) -> list[str]:
    """Names of the schedule templates that make up one local day."""
    evening_out = rng.random() < params["evening_out_probability"]
    if non_workday:
        outing = day.dayofweek != 5 and rng.random() < params["non_workday_outing_probability"]
        names = ["non_workday_out" if outing else "non_workday_home"]
    elif user.has_office and rng.random() >= remote_p:
        names = ["office_day"] + (["evening_out_after_office"] if evening_out else [])
    else:
        third_place = rng.random() < params["third_place_probability"]
        names = ["remote_day_third_place" if third_place else "remote_day_home"]
        names += ["evening_out"] if evening_out else []
    return names + ["night"]


def _day_plan(user, day: pd.Timestamp, non_workday: bool, remote_p: float, params: dict, rng) -> list[tuple]:
    """List of (place_kind, place_ref, start, end) for one local day."""
    plan = []
    for name in _day_template(user, day, non_workday, remote_p, params, rng):
        for kind, start, end in params["schedule"][name]:
            begin = day + pd.Timedelta(hours=float(rng.uniform(*start)))
            finish = day + pd.Timedelta(hours=float(rng.uniform(*end)))
            if kind in ("home", "office"):
                plan.append((kind, None, begin, finish))
            else:
                plan.append(("third_place", rng.choice(getattr(user, f"favourite_{kind}")), begin, finish))
    return plan


def generate_month(world: SyntheticWorld, month: str, params: dict[str, Any], calendar: dict[str, Any]) -> pd.DataFrame:
    """Raw dwells (raw schema) whose UTC timestamp falls in ``month``."""
    rng = np.random.default_rng([params["seed"], int(month)])
    first = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
    days = pd.date_range(first - pd.Timedelta(days=1), first + pd.offsets.MonthEnd(1), freq="D")
    non_workday = dict(zip(days, workdays.flag_non_workday(pd.Series(days), calendar).to_numpy()))
    remote_base = params["remote_day_probability"][month]
    places = world.places.set_index("place_id")
    mid_lat = (params["bbox"]["lat_min"] + params["bbox"]["lat_max"]) / 2

    rows = []
    for user in world.users.itertuples():
        remote_p = min(params["max_remote_day_probability"], remote_base * user.remote_affinity)
        for day in days:
            for kind, ref, start, end in _day_plan(user, day, non_workday[day], remote_p, params, rng):
                if rng.random() > user.observation_rate:
                    continue
                if kind == "home":
                    lat, lon = user.home_lat, user.home_lon
                elif kind == "office":
                    lat, lon = user.office_lat, user.office_lon
                else:
                    lat, lon = places.at[ref, "lat"], places.at[ref, "lon"]
                rows.append((user.identifier, user.identifier_type, kind, ref, lat, lon, start, end))

    dwells = pd.DataFrame(
        rows,
        columns=["identifier", "identifier_type", "place_kind", "place_id", "lat", "lon", "local_start", "local_end"],
    )
    dlat, dlon = _meters_to_degrees(rng.normal(0, params["gps_jitter_m"], (2, len(dwells))), mid_lat)
    utc_offset = pd.Timedelta(hours=params["utc_offset_hours"])
    raw = pd.DataFrame(
        {
            "identifier": dwells["identifier"],
            "identifier_type": dwells["identifier_type"],
            "timestamp": dwells["local_start"] - utc_offset,
            "local_date_time": dwells["local_start"],
            "province": params["province"],
            "classification": np.where(
                rng.random(len(dwells)) < params["non_area_dwell_share"], "STOP", "AREA_DWELL"
            ),
            "duration_seconds": (dwells["local_end"] - dwells["local_start"]).dt.total_seconds(),
            "centroid_latitude": dwells["lat"] + dlat[0],
            "centroid_longitude": dwells["lon"] + dlon[1],
            "bump_count": rng.poisson(params["bump_count_mean"], len(dwells)) + 1,
            "potential_start_time": dwells["local_start"] - utc_offset,
            "potential_end_time": dwells["local_end"] - utc_offset,
            "true_place_kind": dwells["place_kind"],
            "true_place_id": dwells["place_id"],
        }
    )
    in_month = raw["timestamp"].dt.strftime("%Y%m") == month
    return raw[in_month].sort_values(["identifier", "timestamp"]).reset_index(drop=True)


def generate(settings: dict[str, Any]) -> tuple[SyntheticWorld, dict[str, pd.DataFrame]]:
    """World plus raw dwells for every month in ``synthetic.months``."""
    params = settings["synthetic"]
    world = make_world(params)
    months = {month: generate_month(world, month, params, settings["calendar"]) for month in params["months"]}
    return world, months
