"""Export the published aggregate datasets and figures into ``results/``.

Drops geometry (one boundary file is written separately) and rows based on
fewer than ``min_devices_per_row`` devices.

    python tools/export_public_results.py --config configs/public_results.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame(gpd.read_file(path, ignore_geometry=True))


def export(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    source = (config_path.parent / config["source_dir"]).resolve()
    target = (config_path.parent / config["target_dir"]).resolve()

    for item in config["datasets"]:
        frame = _read(source / item["source"])[item["columns"]]
        if item["device_count_column"]:
            frame = frame[frame[item["device_count_column"]] >= config["min_devices_per_row"]]
        out = target / item["target"]
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        print(f"{out.relative_to(target)}: {len(frame):,} rows")

    spec = config["boundaries"]
    areas = gpd.read_file(source / spec["source"])[[*spec["columns"], "geometry"]].drop_duplicates(spec["id_column"])
    areas["geometry"] = (
        areas.to_crs(spec["projected_crs"]).simplify(spec["simplify_tolerance_m"]).to_crs(4326)
    )
    out = target / spec["target"]
    areas.to_file(out, driver="GeoJSON")
    print(f"{out.relative_to(target)}: {len(areas):,} areas")

    for figure in config["figures"]:
        out = target / "figures" / Path(figure).name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / figure, out)
    print(f"figures: {len(config['figures'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    export(parser.parse_args().config)


if __name__ == "__main__":
    main()
