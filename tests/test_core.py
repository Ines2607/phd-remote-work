from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from remote_work_utils import workdays
from remote_work_utils.geo import encode_geohash, geohash_to_polygon
from remote_work_utils.hours import explode_hours
from remote_work_utils.settings import load_settings, resolve_path

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def settings():
    return load_settings(REPO / "configs" / "regions" / "gushdan.json")


@pytest.mark.parametrize(
    "day, expected",
    [
        ("2023-04-02", False),  # Sunday, regular workday
        ("2023-04-05", True),  # Erev Pesach
        ("2023-04-06", True),  # Pesach
        ("2023-04-07", True),  # Friday
        ("2023-04-08", True),  # Saturday
        ("2023-09-15", True),  # Erev Rosh Hashana (Friday)
        ("2023-09-25", True),  # Yom Kippur
        ("2023-09-26", False),  # day after Yom Kippur
        ("2022-11-01", True),  # Knesset election
    ],
)
def test_israeli_non_workdays(settings, day, expected):
    assert bool(workdays.flag_non_workday(pd.Series(pd.to_datetime([day])), settings["calendar"]).iloc[0]) is expected


def test_shabbat_window(settings):
    dates = pd.Series(pd.to_datetime(["2023-04-14", "2023-04-14", "2023-04-15", "2023-04-15"]))
    hours = pd.Series([17, 18, 17, 18])
    shabbat = workdays.in_hour_windows(dates, hours, workdays.hour_window(settings["calendar"], "shabbat"))
    assert shabbat.tolist() == [False, True, True, False]


def test_explode_hours_matches_date_range():
    dwells = pd.DataFrame(
        {
            "start_date_time": pd.to_datetime(["2020-01-01 22:30", "2020-01-02 09:00"]),
            "end_date_time": pd.to_datetime(["2020-01-02 01:10", "2020-01-02 09:05"]),
        }
    )
    hours = explode_hours(dwells)
    assert hours["hour"].tolist() == [22, 23, 0, 1, 9]
    assert hours.index.tolist() == [0, 0, 0, 0, 1]


@pytest.mark.parametrize("swapped", [False, True])
def test_geohash_round_trip(swapped):
    lat, lon = 32.0853, 34.7818
    geohash = encode_geohash([lat], [lon], 7, swapped)[0]
    minx, miny, maxx, maxy = geohash_to_polygon(geohash, swapped).bounds
    assert minx <= lon <= maxx and miny <= lat <= maxy


def test_settings_extends_and_placeholders(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({"paths": {"a": "{raw}/x_{month}.csv"}, "k": {"x": 1, "y": 2}}))
    (tmp_path / "child.json").write_text(json.dumps({"extends": "base.json", "k": {"y": 3}}))
    (tmp_path / "settings.ini").write_text("[Paths]\nraw = /data/raw\nprocessed = /data/processed\n")
    settings = load_settings(tmp_path / "child.json")
    assert settings["k"] == {"x": 1, "y": 3}
    assert resolve_path(settings, "a", month="202001") == Path("/data/raw/x_202001.csv")
