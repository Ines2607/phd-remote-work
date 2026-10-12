"""
Helpers to find candidate third-place locations that look like office-type places
based on monthly word2vec scores per geohash.

This is a notebook-friendly wrapper that can be called directly from notebooks.
"""

import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _init_project_root() -> None:
    """
    Ensure project root is on sys.path and geofunctions is importable.
    """
    project_root = (
        "/Users/inessat/Documents/phd/empirics/israel/notebooks/phd-remote-work"
    )
    if project_root not in sys.path:
        sys.path.append(project_root)


def _import_utils():
    """
    Import geofunctions.utils after sys.path is prepared.
    Separated for easier testing.
    """
    from geofunctions import utils

    return utils


def load_month_scores(
    utils_module,
    input_table: str,
    month: str,
) -> pd.DataFrame:
    """
    Load scores for a specific month from the input BigQuery table.
    """
    # We allow month to be either DATE or STRING in the table.
    query = f"""
    SELECT
      *
    FROM `{input_table}`
    WHERE CAST(month AS STRING) = '{month}'
    """

    df = utils_module.read_query_to_dataframe(query)
    return df


def compute_tail_flags(
    df: pd.DataFrame,
    n_std: float = 2.0,
    outstanding_scores: Iterable[str] | None = None,
    not_outstanding_scores: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Compute high/low tail flags per score and a combined candidate flag using
    **standardized scores (z-scores)**.

    For each score:
      z_score = (value - mean) / std
      high_tail_threshold_z = +n_std
      low_tail_threshold_z  = -n_std

    If outstanding_scores / not_outstanding_scores are not provided, sensible
    defaults are used:
      outstanding_scores     = weekday, morning, early_afternoon, stay_long
      not_outstanding_scores = evening, friday, saturday

    Returns:
        df_with_flags, high_tail_thresholds, low_tail_thresholds
    """
    if outstanding_scores is None:
        outstanding_scores = [
            "score_weekday",
            "score_morning",
            "score_early_afternoon",
            "score_stay_long",
        ]
    else:
        outstanding_scores = list(outstanding_scores)

    if not_outstanding_scores is None:
        not_outstanding_scores = [
            "score_evening",
            "score_friday",
            "score_saturday",
        ]
    else:
        not_outstanding_scores = list(not_outstanding_scores)

    required_cols = [
        "score_home",
        "score_weekday",
        "score_morning",
        "score_early_afternoon",
        "score_evening",
        "score_friday",
        "score_saturday",
        "score_stay_long",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")

    df = df.copy()

    # High tails computed for all scores that can be part of the logic:
    # both "outstanding" (must be high) and "not_outstanding" (must NOT be high).
    high_tail_cols: Iterable[str] = sorted(
        set(list(outstanding_scores) + list(not_outstanding_scores))
    )
    # Low tail for "home" score – we want places that are NOT home-like.
    low_tail_cols: Iterable[str] = ["score_home"]

    high_tail_thresholds: Dict[str, float] = {}
    low_tail_thresholds: Dict[str, float] = {}

    # Compute standardized scores (z-scores) per column.
    # Thresholds are defined in z-space (multiples of std around 0).
    score_cols = list(high_tail_cols) + list(low_tail_cols)
    for col in score_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            mean_val = np.nan
            std_val = np.nan
        else:
            mean_val = float(s.mean())
            std_val = float(s.std())

        if np.isnan(std_val) or std_val == 0:
            z = pd.Series(np.nan, index=df.index)
        else:
            z = (pd.to_numeric(df[col], errors="coerce") - mean_val) / std_val

        # Store standardized value for each score.
        df[f"z_{col}"] = z

        # Thresholds in standardized space.
        if col in high_tail_cols:
            high_tail_thresholds[col] = float(n_std)
        if col in low_tail_cols:
            low_tail_thresholds[col] = float(-n_std)

    # Create boolean flags for each tail using z-scores.
    for col in high_tail_cols:
        thr = high_tail_thresholds[col]
        flag_col = f"is_high_tail_{col}"
        if np.isnan(thr):
            df[flag_col] = False
        else:
            df[flag_col] = df[f"z_{col}"] >= thr

    for col in low_tail_cols:
        thr = low_tail_thresholds[col]
        flag_col = f"is_low_tail_{col}"
        if np.isnan(thr):
            df[flag_col] = False
        else:
            df[flag_col] = df[f"z_{col}"] <= thr

    # Combined candidate flag:
    # - Low home score
    # - High scores on all "outstanding" dimensions
    # - NOT in the high tails for "not_outstanding" dimensions
    def _all_outstanding_high(row) -> bool:
        if not outstanding_scores:
            return True
        return all(row.get(f"is_high_tail_{col}", False) for col in outstanding_scores)

    def _all_not_outstanding_not_high(row) -> bool:
        if not not_outstanding_scores:
            return True
        return all(not row.get(f"is_high_tail_{col}", False) for col in not_outstanding_scores)

    df["is_candidate_all"] = df["is_low_tail_score_home"] & df.apply(
        lambda r: _all_outstanding_high(r) and _all_not_outstanding_not_high(r),
        axis=1,
    )

    # For convenience, count how many high-tail dimensions each row satisfies,
    # but only over the "outstanding" scores (the ones expected to be high).
    high_tail_flag_cols = [
        f"is_high_tail_{c}" for c in outstanding_scores
    ]
    df["n_high_tail_scores"] = df[high_tail_flag_cols].sum(axis=1)

    return df, high_tail_thresholds, low_tail_thresholds


def summarise_score_distributions(
    df: pd.DataFrame,
    high_tail_thresholds: Dict[str, float],
    low_tail_thresholds: Dict[str, float],
    n_std: float,
) -> pd.DataFrame:
    """
    Build a compact table with distribution statistics for each score.

    The table includes (in raw score space):
    - count, mean, std, min, max
    - p10, p25, p50, p75, p90
    - n_std used for z-score thresholds
    - tail thresholds in z-space (high/low, typically +n_std / -n_std)
    - share of observations in high/low tails (based on z-scores)
    """
    score_cols = [
        "score_home",
        "score_weekday",
        "score_morning",
        "score_early_afternoon",
        "score_evening",
        "score_friday",
        "score_saturday",
        "score_stay_long",
    ]

    rows = []
    for col in score_cols:
        s_raw = pd.to_numeric(df[col], errors="coerce")
        s_raw = s_raw.replace([np.inf, -np.inf], np.nan)

        desc = s_raw.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

        high_thr = high_tail_thresholds.get(col, np.nan)
        low_thr = low_tail_thresholds.get(col, np.nan)

        # Fraction in tails (on the non-missing subset), using z-scores.
        z_col = f"z_{col}"
        if z_col in df.columns:
            z = pd.to_numeric(df[z_col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            z = pd.Series(np.nan, index=df.index)

        valid = z.dropna()
        if len(valid) == 0:
            share_high = np.nan
            share_low = np.nan
        else:
            share_high = (
                float((valid >= high_thr).mean())
                if not np.isnan(high_thr)
                else np.nan
            )
            share_low = (
                float((valid <= low_thr).mean())
                if not np.isnan(low_thr)
                else np.nan
            )

        rows.append(
            {
                "score": col,
                "count": float(desc.get("count", np.nan)),
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "min": float(desc.get("min", np.nan)),
                "p10": float(desc.get("10%", np.nan)),
                "p25": float(desc.get("25%", np.nan)),
                "p50": float(desc.get("50%", np.nan)),
                "p75": float(desc.get("75%", np.nan)),
                "p90": float(desc.get("90%", np.nan)),
                "max": float(desc.get("max", np.nan)),
                "n_std": n_std,
                "high_tail_threshold": high_thr,
                "low_tail_threshold": low_thr,
                "share_high_tail": share_high,
                "share_low_tail": share_low,
            }
        )

    return pd.DataFrame(rows)


def find_third_place_candidates_for_month(
    input_table: str,
    month: str,
    n_std: float = 2.0,
    outstanding_scores: Iterable[str] | None = None,
    not_outstanding_scores: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to use directly in notebooks.

    Returns:
        df_with_flags: full table with all rows + tail flags + is_candidate_all
        df_stats:      compact distribution statistics per score
    """
    _init_project_root()
    utils_module = _import_utils()

    df = load_month_scores(
        utils_module=utils_module,
        input_table=input_table,
        month=month,
    )

    if df.empty:
        # Return empty dataframes with expected structure.
        return df, pd.DataFrame()

    df_with_flags, high_thr, low_thr = compute_tail_flags(
        df=df,
        n_std=n_std,
        outstanding_scores=outstanding_scores,
        not_outstanding_scores=not_outstanding_scores,
    )
    df_stats = summarise_score_distributions(
        df=df_with_flags,
        high_tail_thresholds=high_thr,
        low_tail_thresholds=low_thr,
        n_std=n_std,
    )
    return df_with_flags, df_stats


def find_third_place_candidates(
    df: pd.DataFrame,
    months: Iterable[str],
    n_std: float = 2.0,
    outstanding_scores: Iterable[str] | None = None,
    not_outstanding_scores: Iterable[str] | None = None,
    by_feature_type: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Notebook-friendly entry point when you already have a dataframe in memory.

    Parameters
    ----------
    df : pd.DataFrame
        Input table with at least: month, score_* columns, and optionally feature_type.
    months : list-like of str
        List of months to include (compared to df['month'] converted to string).
    n_std : float
        Number of standard deviations for tail detection.
    outstanding_scores : list-like of str, optional
        Scores that must be high-tail (outstanding). If None, use defaults.
    not_outstanding_scores : list-like of str, optional
        Scores that must NOT be high-tail. If None, use defaults.
    by_feature_type : bool
        If True and column 'feature_type' exists, compute std / flags independently
        by (month, feature_type). Otherwise only by month.

    Returns
    -------
    df_with_flags : pd.DataFrame
        Original rows for the requested months with tail flags and is_candidate_all.
    df_stats : pd.DataFrame
        Distribution statistics per score and per group (month, and optionally feature_type).
    """
    if "month" not in df.columns:
        raise ValueError("Input dataframe must contain a 'month' column.")

    months_str = {str(m) for m in months}

    df = df.copy()
    df["month_str"] = df["month"].astype(str)
    df = df[df["month_str"].isin(months_str)]

    if df.empty:
        return df, pd.DataFrame()

    group_cols = ["month_str"]
    if by_feature_type and "feature_type" in df.columns:
        group_cols.append("feature_type")

    grouped = df.groupby(group_cols, dropna=False)

    df_with_flags_list = []
    stats_list = []

    for keys, g in grouped:
        g_with_flags, high_thr, low_thr = compute_tail_flags(
            df=g,
            n_std=n_std,
            outstanding_scores=outstanding_scores,
            not_outstanding_scores=not_outstanding_scores,
        )
        df_with_flags_list.append(g_with_flags)

        stats = summarise_score_distributions(
            df=g_with_flags,
            high_tail_thresholds=high_thr,
            low_tail_thresholds=low_thr,
            n_std=n_std,
        )

        # Attach group identifiers to the stats table.
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                stats[col] = val
        else:
            stats[group_cols[0]] = keys

        stats_list.append(stats)

    df_with_flags_all = pd.concat(df_with_flags_list, ignore_index=True)
    df_stats_all = pd.concat(stats_list, ignore_index=True)

    return df_with_flags_all, df_stats_all


def compute_tail_flags_from_standardized(
    df: pd.DataFrame,
    n_std: float = 2.0,
    outstanding_scores: Iterable[str] | None = None,
    not_outstanding_scores: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Compute high/low tail flags per score and a combined candidate flag using
    **already standardized scores (z-scores)** that are expected to exist in
    the dataframe.

    This function is similar to compute_tail_flags, but assumes z-score columns
    (e.g., z_score_weekday, z_score_home) already exist in the dataframe.

    For each score:
      high_tail_threshold_z = +n_std
      low_tail_threshold_z  = -n_std

    If outstanding_scores / not_outstanding_scores are not provided, sensible
    defaults are used:
      outstanding_scores     = weekday, morning, early_afternoon, stay_long
      not_outstanding_scores = evening, friday, saturday

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with z-score columns (e.g., z_score_weekday,
        z_score_home).
    n_std : float
        Number of standard deviations for tail detection.
    outstanding_scores : list-like of str, optional
        Scores that must be high-tail (outstanding). If None, use defaults.
    not_outstanding_scores : list-like of str, optional
        Scores that must NOT be high-tail. If None, use defaults.

    Returns
    -------
    df_with_flags : pd.DataFrame
        Original dataframe with tail flags and is_candidate_all added.
    high_tail_thresholds : Dict[str, float]
        High tail thresholds in z-space for each score.
    low_tail_thresholds : Dict[str, float]
        Low tail thresholds in z-space for each score.
    """
    if outstanding_scores is None:
        outstanding_scores = [
            "score_weekday",
            "score_morning",
            "score_early_afternoon",
            "score_stay_long",
        ]
    else:
        outstanding_scores = list(outstanding_scores)

    if not_outstanding_scores is None:
        not_outstanding_scores = [
            "score_evening",
            "score_friday",
            "score_saturday",
        ]
    else:
        not_outstanding_scores = list(not_outstanding_scores)

    df = df.copy()

    # High tails computed for all scores that can be part of the logic:
    # both "outstanding" (must be high) and "not_outstanding"
    # (must NOT be high).
    high_tail_cols: Iterable[str] = sorted(
        set(list(outstanding_scores) + list(not_outstanding_scores))
    )
    # Low tail for "home" score – we want places that are NOT home-like.
    low_tail_cols: Iterable[str] = ["score_home"]

    high_tail_thresholds: Dict[str, float] = {}
    low_tail_thresholds: Dict[str, float] = {}

    # Check that z-score columns exist and use them directly.
    score_cols = list(high_tail_cols) + list(low_tail_cols)
    missing_z_cols = []
    for col in score_cols:
        z_col = f"z_{col}"
        if z_col not in df.columns:
            missing_z_cols.append(z_col)
        else:
            # Thresholds in standardized space.
            if col in high_tail_cols:
                high_tail_thresholds[col] = float(n_std)
            if col in low_tail_cols:
                low_tail_thresholds[col] = float(-n_std)

    if missing_z_cols:
        raise ValueError(
            f"Missing required z-score columns in input data: "
            f"{missing_z_cols}. Expected columns like z_score_weekday, "
            "z_score_home, etc."
        )

    # Create boolean flags for each tail using existing z-scores.
    for col in high_tail_cols:
        thr = high_tail_thresholds[col]
        flag_col = f"is_high_tail_{col}"
        z_col = f"z_{col}"
        if np.isnan(thr):
            df[flag_col] = False
        else:
            z = pd.to_numeric(df[z_col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            df[flag_col] = z >= thr

    for col in low_tail_cols:
        thr = low_tail_thresholds[col]
        flag_col = f"is_low_tail_{col}"
        z_col = f"z_{col}"
        if np.isnan(thr):
            df[flag_col] = False
        else:
            z = pd.to_numeric(df[z_col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            df[flag_col] = z <= thr

    # Combined candidate flag:
    # - Low home score
    # - High scores on all "outstanding" dimensions
    # - NOT in the high tails for "not_outstanding" dimensions
    def _all_outstanding_high(row) -> bool:
        if not outstanding_scores:
            return True
        return all(
            row.get(f"is_high_tail_{col}", False)
            for col in outstanding_scores
        )

    def _all_not_outstanding_not_high(row) -> bool:
        if not not_outstanding_scores:
            return True
        return all(
            not row.get(f"is_high_tail_{col}", False)
            for col in not_outstanding_scores
        )

    df["is_candidate_all"] = df["is_low_tail_score_home"] & df.apply(
        lambda r: (
            _all_outstanding_high(r) and _all_not_outstanding_not_high(r)
        ),
        axis=1,
    )

    # For convenience, count how many high-tail dimensions each row satisfies,
    # but only over the "outstanding" scores (the ones expected to be high).
    high_tail_flag_cols = [
        f"is_high_tail_{c}" for c in outstanding_scores
    ]
    df["n_high_tail_scores"] = df[high_tail_flag_cols].sum(axis=1)

    return df, high_tail_thresholds, low_tail_thresholds


def find_third_place_candidates_from_standardized(
    df: pd.DataFrame,
    months: Iterable[str],
    n_std: float = 2.0,
    outstanding_scores: Iterable[str] | None = None,
    not_outstanding_scores: Iterable[str] | None = None,
    by_feature_type: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Notebook-friendly entry point when you already have a standardized
    dataframe (with z-score columns) in memory.

    This function is similar to find_third_place_candidates, but works with
    dataframes that already have z-score columns computed (e.g.,
    z_score_weekday, z_score_home, etc.). It skips the standardization step
    and uses the existing z-scores directly.

    Parameters
    ----------
    df : pd.DataFrame
        Input table with at least: month, z_score_* columns (e.g.,
        z_score_weekday, z_score_home), and optionally feature_type. Raw score
        columns (score_*) should also be present for statistics.
    months : list-like of str
        List of months to include (compared to df['month'] converted to
        string).
    n_std : float
        Number of standard deviations for tail detection.
    outstanding_scores : list-like of str, optional
        Scores that must be high-tail (outstanding). If None, use defaults.
    not_outstanding_scores : list-like of str, optional
        Scores that must NOT be high-tail. If None, use defaults.
    by_feature_type : bool
        If True and column 'feature_type' exists, compute flags independently
        by (month, feature_type). Otherwise only by month.

    Returns
    -------
    df_with_flags : pd.DataFrame
        Original rows for the requested months with tail flags and is_candidate_all.
    df_stats : pd.DataFrame
        Distribution statistics per score and per group (month, and optionally feature_type).
    """
    if "month" not in df.columns:
        raise ValueError("Input dataframe must contain a 'month' column.")

    months_str = {str(m) for m in months}

    df = df.copy()
    df["month_str"] = df["month"].astype(str)
    df = df[df["month_str"].isin(months_str)]

    if df.empty:
        return df, pd.DataFrame()

    group_cols = ["month_str"]
    if by_feature_type and "feature_type" in df.columns:
        group_cols.append("feature_type")

    grouped = df.groupby(group_cols, dropna=False)

    df_with_flags_list = []
    stats_list = []

    for keys, g in grouped:
        g_with_flags, high_thr, low_thr = compute_tail_flags_from_standardized(
            df=g,
            n_std=n_std,
            outstanding_scores=outstanding_scores,
            not_outstanding_scores=not_outstanding_scores,
        )
        df_with_flags_list.append(g_with_flags)

        stats = summarise_score_distributions(
            df=g_with_flags,
            high_tail_thresholds=high_thr,
            low_tail_thresholds=low_thr,
            n_std=n_std,
        )

        # Attach group identifiers to the stats table.
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                stats[col] = val
        else:
            stats[group_cols[0]] = keys

        stats_list.append(stats)

    df_with_flags_all = pd.concat(df_with_flags_list, ignore_index=True)
    df_stats_all = pd.concat(stats_list, ignore_index=True)

    return df_with_flags_all, df_stats_all


def plot_kde_for_scores(
    df: pd.DataFrame,
    scores: Iterable[str] | None = None,
    figsize: tuple[float, float] = (6, 4),
) -> None:
    """
    Plot KDE diagrams for each score column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with score columns.
    scores : list-like of str, optional
        Which score columns to plot. If None, uses the default set of scores.
    figsize : tuple
        Figure size for each individual plot.
    """
    if scores is None:
        scores = [
            "score_home",
            "score_weekday",
            "score_morning",
            "score_early_afternoon",
            "score_evening",
            "score_friday",
            "score_saturday",
            "score_stay_long",
        ]

    for col in scores:
        if col not in df.columns:
            continue
        data = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if data.empty:
            continue

        plt.figure(figsize=figsize)
        sns.kdeplot(data, fill=True)
        plt.title(f"KDE for {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.show()
