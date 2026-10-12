"""Expand dwells to one row per clock hour they touch."""

from __future__ import annotations

import numpy as np
import pandas as pd

ONE_HOUR = pd.Timedelta(hours=1)


def explode_hours(dwells: pd.DataFrame) -> pd.DataFrame:
    """Repeat each dwell once per hour in ``[floor(start), ceil(end))``.

    Adds ``hours_in_interval`` (hour start timestamp) and ``hour`` (0-23).
    Row order and index follow the input (dwell by dwell, hours ascending).
    A dwell that touches no full hour keeps one row with a missing hour.
    """
    first_hour = dwells["start_date_time"].dt.floor("h")
    last_hour = dwells["end_date_time"].dt.ceil("h") - ONE_HOUR
    n_hours = ((last_hour - first_hour) // ONE_HOUR + 1).clip(lower=0).astype(int).to_numpy()
    repeats = np.maximum(n_hours, 1)

    hours = dwells.loc[dwells.index.repeat(repeats)].copy()
    offsets = np.arange(repeats.sum()) - np.repeat(np.cumsum(repeats) - repeats, repeats)
    hour_starts = np.repeat(first_hour.to_numpy(), repeats) + offsets * np.timedelta64(1, "h")
    hours["hours_in_interval"] = pd.Series(hour_starts, index=hours.index).where(
        np.repeat(n_hours > 0, repeats)
    )
    hours["hour"] = hours["hours_in_interval"].dt.hour
    return hours
