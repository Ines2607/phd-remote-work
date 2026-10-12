"""Generate the synthetic demo inputs (raw dwells, stat areas, boundary, places).

    python pipelines/make_demo_data.py --config configs/demo.json
"""

from __future__ import annotations

import argparse

from remote_work_utils.settings import load_settings, resolve_path
from remote_work_utils.synthetic import generate


def write_demo_data(settings: dict) -> None:
    world, months = generate(settings)
    for key, frame in [("stat_areas", world.stat_areas), ("boundary", world.boundary)]:
        path = resolve_path(settings, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_file(path, driver="GeoJSON")
    world.places.to_csv(resolve_path(settings, "places"), index=False)
    world.users.to_csv(resolve_path(settings, "synthetic_users"), index=False)
    for month, raw in months.items():
        path = resolve_path(settings, "dwells_raw", month=month)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(path, index=False)
        print(f"{month}: {len(raw):,} dwells, {raw['identifier'].nunique()} users -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    write_demo_data(load_settings(parser.parse_args().config))


if __name__ == "__main__":
    main()
