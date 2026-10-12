"""Run home -> work -> remote-work detection for one or more months.

Example::

    python pipelines/run_month.py --config configs/demo.json --month 202004
    python pipelines/run_month.py --config configs/regions/gushdan.json \
        --month 202001 202003 --steps home work remote
"""

from __future__ import annotations

import argparse

from remote_work_utils.pipeline import STEPS, run_month
from remote_work_utils.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Settings JSON (see configs/)")
    parser.add_argument("--month", required=True, nargs="+", help="One or more months, YYYYMM")
    parser.add_argument("--steps", nargs="+", default=STEPS[1:], choices=STEPS)
    args = parser.parse_args()

    settings = load_settings(args.config)
    for month in args.month:
        result = run_month(settings, month, steps=args.steps)
        print(month, result.metrics)


if __name__ == "__main__":
    main()
