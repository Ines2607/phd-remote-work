"""Workday calendar: weekends, public holidays, holiday eves and extra dates.

All values come from the ``calendar`` settings section, e.g. for Israel::

    "calendar": {
      "weekend_days": [4, 5],                     # pandas weekday: Fri, Sat
      "holiday_calendar": {"country": "IL", "include_eve": true,
                           "include_dates": [...election days...],
                           "exclude_dates": []},
      "hour_windows": {"night": [...], "shabbat": [...]}
    }

Hour windows are lists of ``{"start_hour", "end_hour", "weekdays"?}`` with
``start_hour <= hour < end_hour``; ``weekdays`` (optional) refers to the
weekday of the dwell date.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd


def public_holidays(calendar: dict[str, Any], years: Iterable[int]) -> set[date]:
    """Public holidays of ``holiday_calendar.country`` for the given years."""
    import holidays

    holiday_settings = calendar["holiday_calendar"]
    provider = holidays.country_holidays(
        holiday_settings["country"],
        subdiv=holiday_settings.get("subdivision"),
        years=sorted({int(year) for year in years}),
    )
    return set(provider.keys())


def non_workday_dates(calendar: dict[str, Any], years: Iterable[int]) -> set[date]:
    """Holidays, their eves (if ``include_eve``) and ``include_dates``,
    minus ``exclude_dates``. Weekends are handled separately."""
    holiday_settings = calendar["holiday_calendar"]
    holiday_dates = public_holidays(calendar, years)
    result = set(holiday_dates)
    if holiday_settings["include_eve"]:
        result.update(day - timedelta(days=1) for day in holiday_dates)
    result.update(pd.to_datetime(holiday_settings["include_dates"]).date)
    result.difference_update(pd.to_datetime(holiday_settings["exclude_dates"]).date)
    return result


def years_of(dates: pd.Series) -> set[int]:
    return set(pd.to_datetime(dates).dt.year.dropna().astype(int).unique())


def flag_weekend_only(dates: pd.Series, calendar: dict[str, Any]) -> pd.Series:
    """True on configured weekend days (no holidays)."""
    return pd.to_datetime(dates).dt.weekday.isin(calendar["weekend_days"])


def flag_non_workday(dates: pd.Series, calendar: dict[str, Any]) -> pd.Series:
    """True on weekends, holidays, holiday eves and extra non-workdays."""
    dates = pd.to_datetime(dates)
    special = non_workday_dates(calendar, years_of(dates))
    return dates.dt.weekday.isin(calendar["weekend_days"]) | dates.dt.date.isin(special)


def flag_holiday(dates: pd.Series, calendar: dict[str, Any]) -> pd.Series:
    """True on public holidays only (no eves, no extra dates)."""
    dates = pd.to_datetime(dates)
    return dates.dt.date.isin(public_holidays(calendar, years_of(dates)))


def in_hour_windows(
    dates: pd.Series, hours: pd.Series, windows: list[dict[str, Any]]
) -> pd.Series:
    """True where ``hour`` falls in any window (optionally restricted by weekday)."""
    weekday = pd.to_datetime(dates).dt.dayofweek
    selected = pd.Series(False, index=hours.index)
    for window in windows:
        in_window = (hours >= window["start_hour"]) & (hours < window["end_hour"])
        if "weekdays" in window:
            in_window &= weekday.isin(window["weekdays"])
        selected |= in_window
    return selected


def hour_window(calendar: dict[str, Any], name: str) -> list[dict[str, Any]]:
    try:
        return calendar["hour_windows"][name]
    except KeyError as exc:
        raise KeyError(f"calendar.hour_windows.{name} is not defined") from exc
