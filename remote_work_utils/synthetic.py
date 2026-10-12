"""Synthetic GPS dwells with known home, office and third places.

Used by the demo notebooks and tests. Nothing here comes from real devices:
users, places and visits are drawn from the ``synthetic`` settings section.

Behaviour per user and day (local time, workday calendar from settings):

* nights and Shabbat / holidays at home;
* office workers go to the office on workdays, or stay home on remote days
  (probability ``remote_day_probability[month] * personal affinity``), with
  occasional third-place visits in the day or evening;
* users without an office spend workdays at home or at third places;
* each dwell is observed with a user-specific probability (data sparsity).
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
THIRD_PLACE_CATEGORIES = ["cafe", "coworking", "library", "gym", "park", "restaurant", "bar", "mall"]
WORK_FRIENDLY_CATEGORIES = {"cafe", "coworking", "library"}


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

    n_hubs = max(3, n_users // 40)
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
        dlat, dlon = _meters_to_degrees(rng.normal(0, 150, 2), mid_lat)
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
                rng.normal(params["observation_probability"], 0.15, n_users), 0.35, 0.98
            ),
        }
    )

    n_places = params["n_third_places"]
    places = pd.DataFrame(
        {
            "place_id": [f"place-{i:03d}" for i in range(n_places)],
            "category": rng.choice(THIRD_PLACE_CATEGORIES, size=n_places),
            "lat": rng.uniform(bbox["lat_min"], bbox["lat_max"], n_places),
            "lon": rng.uniform(bbox["lon_min"], bbox["lon_max"], n_places),
        }
    )
    favourites = []
    for user in users.itertuples():
        distance = _distance_m(user.home_lat, user.home_lon, places["lat"], places["lon"])
        nearest = np.argsort(distance)[:8]
        favourites.append(list(places["place_id"].to_numpy()[rng.choice(nearest, size=3, replace=False)]))
    users["favourite_places"] = favourites

    boundary = gpd.GeoDataFrame(
        {"name": ["synthetic boundary"]},
        geometry=[box(bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]).buffer(0.01)],
        crs=4326,
    )
    return SyntheticWorld(users=users, places=places, stat_areas=stat_areas, boundary=boundary)


def _day_plan(user, day: pd.Timestamp, non_workday: bool, remote_p: float, params: dict, rng) -> list[tuple]:
    """List of (place_kind, place_ref, start, end) for one local day."""
    h = lambda hours: day + pd.Timedelta(hours=float(hours))  # noqa: E731
    plan = []
    evening_out = rng.random() < params["evening_out_probability"]
    favourite = lambda: rng.choice(user.favourite_places)  # noqa: E731
    is_shabbat = day.dayofweek == 5

    if non_workday:
        plan.append(("home", None, h(rng.uniform(7.5, 9.5)), h(rng.uniform(12, 14))))
        if not is_shabbat and rng.random() < 0.6:
            plan.append(("third_place", favourite(), h(rng.uniform(14, 15)), h(rng.uniform(16, 18))))
            plan.append(("home", None, h(rng.uniform(18, 19)), h(rng.uniform(20.5, 21.5))))
        else:
            plan.append(("home", None, h(rng.uniform(14, 15)), h(rng.uniform(20.5, 21.5))))
    elif user.has_office and rng.random() >= remote_p:
        plan.append(("office", None, h(rng.uniform(8.3, 9.5)), h(rng.uniform(12.3, 13))))
        plan.append(("office", None, h(rng.uniform(13.5, 14)), h(rng.uniform(17, 18.5))))
        if evening_out:
            plan.append(("third_place", favourite(), h(rng.uniform(19, 19.5)), h(rng.uniform(20.5, 21.5))))
    else:
        if rng.random() < params["third_place_probability"]:
            plan.append(("home", None, h(rng.uniform(7.5, 8.5)), h(rng.uniform(11, 12))))
            plan.append(("third_place", favourite(), h(rng.uniform(12.5, 13)), h(rng.uniform(14.5, 15.5))))
            plan.append(("home", None, h(rng.uniform(16, 16.5)), h(rng.uniform(18.5, 19.5))))
        else:
            plan.append(("home", None, h(rng.uniform(7.5, 8.5)), h(rng.uniform(12.5, 13.5))))
            plan.append(("home", None, h(rng.uniform(13.6, 14)), h(rng.uniform(18.5, 19.5))))
        if evening_out:
            plan.append(("third_place", favourite(), h(rng.uniform(19.8, 20.2)), h(rng.uniform(21, 21.8))))
    plan.append(("home", None, h(rng.uniform(22, 23.5)), h(24 + rng.uniform(6.5, 8))))
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
        remote_p = min(0.95, remote_base * user.remote_affinity)
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
            "province": "Tel Aviv",
            "classification": np.where(
                rng.random(len(dwells)) < params["non_area_dwell_share"], "STOP", "AREA_DWELL"
            ),
            "duration_seconds": (dwells["local_end"] - dwells["local_start"]).dt.total_seconds(),
            "centroid_latitude": dwells["lat"] + dlat[0],
            "centroid_longitude": dwells["lon"] + dlon[1],
            "bump_count": rng.poisson(3, len(dwells)) + 1,
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
