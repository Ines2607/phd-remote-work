"""Confidence intervals and group-comparison charts used in the demos and paper figures.

Shares get 95% Wilson binomial intervals, means a 95% normal interval.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

Z_95 = 1.96


def add_wilson_ci(df: pd.DataFrame, count_col: str = "k", n_col: str = "n", z: float = Z_95) -> pd.DataFrame:
    n = df[n_col].astype(float)
    k = df[count_col].astype(float)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return df.assign(share=p, ci_low=(center - half).clip(lower=0), ci_high=(center + half).clip(upper=1))


def share_by(df: pd.DataFrame, group_cols: Sequence[str], flag_col: str) -> pd.DataFrame:
    """Share of rows with ``flag_col`` per group, with Wilson CI."""
    grouped = df.groupby(list(group_cols), observed=True).agg(n=(flag_col, "size"), k=(flag_col, "sum")).reset_index()
    return add_wilson_ci(grouped)


def mean_by(df: pd.DataFrame, group_cols: Sequence[str], value_col: str, z: float = Z_95) -> pd.DataFrame:
    """Mean of ``value_col`` per group, with normal CI."""
    grouped = (
        df.dropna(subset=[value_col])
        .groupby(list(group_cols), observed=True)[value_col]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )
    se = grouped["sd"] / np.sqrt(grouped["n"])
    return grouped.assign(ci_low=grouped["mean"] - z * se, ci_high=grouped["mean"] + z * se)


def plot_ci_dots(ax, table: pd.DataFrame, label_col: str, value_col: str, overall: float | None = None, scale: float = 1.0):
    """Horizontal dots with error bars, one row per label (top to bottom in table order)."""
    y = np.arange(len(table))
    value = table[value_col] * scale
    ax.errorbar(
        value,
        y,
        xerr=[(value - table["ci_low"] * scale).clip(lower=0), (table["ci_high"] * scale - value).clip(lower=0)],
        fmt="o",
        ms=5,
        color="#2b6cb0",
        ecolor="#2b6cb0",
        elinewidth=1.4,
        capsize=2.5,
    )
    if overall is not None:
        ax.axvline(overall * scale, ls="--", lw=1.2, color="#dd6b20", label=f"All rows ({overall * scale:.1f})")
        ax.legend(fontsize=8, loc="best")
    ax.set_yticks(y)
    ax.set_yticklabels(table[label_col])
    ax.set_ylim(len(table) - 0.5, -0.5)
    return ax
