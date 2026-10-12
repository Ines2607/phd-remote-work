import pandas as pd

from remote_work_utils import dwell_stats
from remote_work_utils.settings import load_settings

SETTINGS = load_settings("configs/template.json")


def test_monthly_stats_use_israeli_calendar():
    local = pd.to_datetime(["2023-03-05 09:00", "2023-03-10 09:00", "2023-03-11 20:00", "2023-03-12 09:00"])
    raw = pd.DataFrame(
        {
            "identifier": ["a", "a", "b", "b"],
            "identifier_type": ["GAID", "GAID", "IDFA", "IDFA"],
            "local_date_time": local,
            "duration_seconds": [3600, 7200, 18000, 100],
            "centroid_latitude": [32.08] * 4,
            "centroid_longitude": [34.78] * 4,
        }
    )
    stats = dwell_stats.monthly_dwell_stats(raw, "202303", SETTINGS)
    assert stats["total_users"] == 2
    # Friday 10 March and Saturday 11 March are non-workdays.
    assert stats["share_dwells_non_workday"] == 0.5
    assert stats["share_dwells_over_min_duration"] == 0.75
    assert stats["share_dwells_day"] == 0.75


def test_outlier_month_is_flagged():
    monthly = pd.DataFrame({"month": [f"2020{m:02d}" for m in range(1, 7)], "total_users": [100, 101, 99, 100, 102, 0]})
    result = dwell_stats.outlier_months(monthly, SETTINGS["dwell_stats"]["outliers"])
    assert result["remove"].tolist() == [False] * 5 + [True]
