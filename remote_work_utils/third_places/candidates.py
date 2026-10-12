"""Third-place candidates from standardised behaviour scores (z-score tails).

A place is a candidate when its ``low_scores`` (e.g. home-likeness) are in the
low tail, all ``outstanding_scores`` (weekday, morning, long stays) are in the
high tail, and none of the ``not_outstanding_scores`` (evening, weekend) are.
Tails are ``|z| >= n_std`` within each group (e.g. month).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def scores_from_visit_stats(stats: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add ``score_*`` columns from :func:`poi_visit_stats` columns (``score_columns`` mapping)."""
    return stats.assign(**{score: stats[column] for score, column in params["score_columns"].items()})


def standardise(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add ``z_<column>`` (missing if the column has no variance)."""
    out = df.copy()
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        std = values.std()
        out[f"z_{column}"] = (values - values.mean()) / std if std and not np.isnan(std) else np.nan
    return out


def tail_flags(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Tail flags on existing ``z_*`` columns and the combined ``is_candidate_all``."""
    n_std = params["n_std"]
    out = df.copy()
    high = sorted(set(params["outstanding_scores"]) | set(params["not_outstanding_scores"]))
    for column in high:
        out[f"is_high_tail_{column}"] = out[f"z_{column}"] >= n_std
    for column in params["low_scores"]:
        out[f"is_low_tail_{column}"] = out[f"z_{column}"] <= -n_std
    low_ok = out[[f"is_low_tail_{c}" for c in params["low_scores"]]].all(axis=1)
    outstanding = out[[f"is_high_tail_{c}" for c in params["outstanding_scores"]]]
    not_outstanding = out[[f"is_high_tail_{c}" for c in params["not_outstanding_scores"]]]
    out["is_candidate_all"] = low_ok & outstanding.all(axis=1) & ~not_outstanding.any(axis=1)
    out["n_high_tail_scores"] = outstanding.sum(axis=1)
    return out


def find_candidates(df: pd.DataFrame, params: dict[str, Any], group: list[str] | None = None) -> pd.DataFrame:
    """Standardise scores within each group and flag candidates."""
    scores = sorted(set(params["outstanding_scores"]) | set(params["not_outstanding_scores"]) | set(params["low_scores"]))
    missing = [c for c in scores if c not in df.columns]
    if missing:
        raise KeyError(f"Missing score columns: {missing}")
    if not group:
        return tail_flags(standardise(df, scores), params)
    parts = [tail_flags(standardise(part, scores), params) for _, part in df.groupby(group, dropna=False)]
    return pd.concat(parts, ignore_index=True)


def score_summary(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Distribution of each score and the share of rows in its tail."""
    rows = []
    for column in sorted(set(params["outstanding_scores"]) | set(params["not_outstanding_scores"]) | set(params["low_scores"])):
        values = pd.to_numeric(df[column], errors="coerce")
        z = df[f"z_{column}"]
        rows.append(
            {
                "score": column,
                **values.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                "share_high_tail": float((z >= params["n_std"]).mean()),
                "share_low_tail": float((z <= -params["n_std"]).mean()),
            }
        )
    return pd.DataFrame(rows)
