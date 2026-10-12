"""The package reproduces the original paper scripts on synthetic data.

Runs ``research/03-remote-work/legacy/00{1,2,3}-*.py`` (the scripts used for the
paper) in a temporary folder layout they expect, then runs
``remote_work_utils`` on the same filtered dwells and compares the outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from remote_work_utils.home_work.home import detect_homes
from remote_work_utils.home_work.work import detect_work
from remote_work_utils.prepare import prepare_dwells
from remote_work_utils.remote_work.hourly import build_hourly_panel
from remote_work_utils.remote_work.metrics import remote_work_month
from remote_work_utils.schema import read_dwells
from remote_work_utils.settings import load_settings
from remote_work_utils.synthetic import generate_month, make_world

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "research" / "03-remote-work" / "legacy"
MONTHS = ["201909", "202004"]
LEGACY_CENSUS_COLUMNS = {
    "stat_area_id": "YISHUV_STAT_2022",
    "city_name": "SHEM_YISHUV_ENG",
    "population": "pop_approx",
}

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def legacy_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy")
    raw_dir = root / "raw" / "data_dwells"
    processed = root / "processed"
    for folder in [raw_dir, processed / "data_census", root / "nb", root / "results/diagrams/home-work-hours-plots"]:
        folder.mkdir(parents=True)
    for sub in ["home", "work", "remote_work", "samples"]:
        (processed / "data_dwells" / sub).mkdir(parents=True)
    (root / "settings.ini").write_text(f"[Paths]\nraw = {root / 'raw'}\nprocessed = {processed}\n")

    config = {
        "extends": str(REPO / "configs" / "regions" / "gushdan.json"),
        "paths": {
            "dwells": str(raw_dir / "data_filtered_{month}.parquet"),
            "stat_areas": str(processed / "data_census" / "census_gushdan_main_features.geojson"),
            "output_dir": str(root / "package"),
        },
        "remote_work": {"output_all_variants": True},
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    settings = load_settings(config_path)

    demo = load_settings(REPO / "configs" / "demo.json")
    synthetic = {
        **demo["synthetic"],
        "n_users": 120,
        "months": MONTHS,
        "remote_day_probability": {"201909": 0.1, "202004": 0.6},
    }
    world = make_world(synthetic)
    census = world.stat_areas.rename(columns=LEGACY_CENSUS_COLUMNS)
    census["WrkOutLoc_pcnt"] = np.linspace(10, 60, len(census))
    census.to_file(processed / "data_census" / "census_gushdan_main_features.geojson", driver="GeoJSON")

    env = {**os.environ, "PYTHONPATH": str(REPO), "MPLBACKEND": "Agg"}
    for month in MONTHS:
        raw = generate_month(world, month, synthetic, settings["calendar"])
        filtered = prepare_dwells(raw, {**settings, "prepare": {**settings["prepare"], "clip_to_boundary": False}})
        filtered.to_parquet(raw_dir / f"data_filtered_{month}.parquet", index=False)
        for script in sorted(LEGACY.glob("00*.py")):
            subprocess.run(
                [sys.executable, "-W", "ignore", str(script), "--month", month],
                cwd=root / "nb",
                env=env,
                check=True,
                capture_output=True,
            )
    return settings, processed / "data_dwells"


@pytest.mark.parametrize("month", MONTHS)
def test_outputs_match_legacy(legacy_run, month):
    settings, legacy_dir = legacy_run
    dwells = read_dwells(Path(settings["paths"]["dwells"].format(month=month)), settings)

    homes = detect_homes(dwells, settings).homes
    legacy_homes = pd.read_csv(legacy_dir / f"home/home_geohashes_night_shabbat_{month}.csv", index_col=0)
    pd.testing.assert_frame_equal(
        homes.sort_values("identifier").reset_index(drop=True)[legacy_homes.columns],
        legacy_homes.sort_values("identifier").reset_index(drop=True),
        check_dtype=False,
    )

    work = detect_work(dwells, homes, settings)
    legacy_work = pd.read_csv(legacy_dir / f"work/users_work_home_{month}.csv", index_col=0)
    compare = ["identifier", "geohash_home", "geohash_work", "date", "work_date", "share_days_work_location", "distance_home"]
    pd.testing.assert_frame_equal(
        work.sort_values("identifier").reset_index(drop=True)[compare],
        legacy_work.sort_values("identifier").reset_index(drop=True)[compare],
        check_dtype=False,
    )

    hours = build_hourly_panel(dwells, work, settings, month)
    users, _ = remote_work_month(hours, settings)
    legacy_users = pd.read_csv(legacy_dir / f"remote_work/user_remote_work_{month}.csv", index_col=0)
    pd.testing.assert_frame_equal(
        users.sort_values("identifier").reset_index(drop=True)[legacy_users.columns],
        legacy_users.sort_values("identifier").reset_index(drop=True),
        check_dtype=False,
    )
    assert len(legacy_homes) > 80 and len(legacy_work) > 50 and len(legacy_users) > 40
    assert users["flag_remote_work_smoothed_max_thresh"].between(0.01, 0.99).any()
