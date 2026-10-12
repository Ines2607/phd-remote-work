"""Helpers for 05-neighborhood-retention.ipynb (no SQL)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from cbps import CBPS, balance
except ImportError:  # optional; only run_cbgps_did needs it
    CBPS = balance = None

_NS = None
_MISSING = object()

_ALLOWED_INT = (
    "tp_intensity", "cand_intensity", "atm_intensity", "share_tp",
    "atm_only_intensity", "comm_cand_intensity", "noncomm_cand_intensity",
    "nontp_candidate_intensity",
)
GPS_X_LABELS = {
    "eshkol_madad2019": "2019 SES cluster",
    "age20_59_pcnt": "2019 age 20–59 share",
    "dist_to_cbd_km": "Distance to CBD (km)",
    "amenities_1km_area_wght": "Amenity weight 1 km",
    "share_local_poi_h_pre": "Jan 2020 local | all POIs, hours",
    "wfh_avg_pre": "Jan 2020 WFH share (SA mean)",
    "nontp_candidate_intensity": "Non-TP candidate intensity",
}

_INTENSITY_KEYS = (
    "share_tp", "share_detected", "tp_intensity",
    "cand_intensity", "atm_intensity", "atm_only_intensity",
    "comm_cand_intensity", "noncomm_cand_intensity", "nontp_candidate_intensity",
    "tp_bin", "ipw_weight",
)


def bind(ns):
    """Point helpers at the notebook namespace (constants + write-back)."""
    global _NS
    _NS = ns


def _cfg(name, default=_MISSING):
    """Read `name` from the bound notebook. Never from module globals."""
    if _NS is None:
        raise RuntimeError(
            "Helpers are not bound. After imports run: "
            "import third_places_neighborhood_retention as tpr; tpr.bind(globals())"
        )
    if name in _NS:
        return _NS[name]
    if default is not _MISSING:
        return default
    raise KeyError(
        f"{name!r} is not defined in the notebook. "
        "Define it before calling this helper, or pass it as an argument."
    )


def _write(**kwargs):
    if _NS is None:
        raise RuntimeError("Helpers are not bound; cannot write results back.")
    _NS.update(kwargs)


def _did_window_names(present):
    present = {str(w) for w in present}
    names = list(_cfg("DID_WINDOWS"))
    return [w for w in names if w in present]


def apply_treat_flag(sa, flag_col=None, rule=None):
    flag_col = _cfg("TREAT_FLAG") if flag_col is None else flag_col
    rule = _cfg("treat_rule") if rule is None else rule
    out = sa.copy()
    out[flag_col] = pd.Series(rule(out), index=out.index).fillna(0).astype(int)
    if out[flag_col].nunique() < 2:
        raise ValueError(f"{flag_col} is constant after treat_rule — check the rule")
    return out

def standardised_mean_diff(treated, control):
    """SMD for continuous/binary; uses pooled SD."""
    mt, mc = treated.mean(), control.mean()
    st, sc = treated.std(ddof=1), control.std(ddof=1)
    nt, nc = len(treated), len(control)
    pooled = np.sqrt(((nt - 1) * st**2 + (nc - 1) * sc**2) / max(nt + nc - 2, 1))
    if pooled == 0 or np.isnan(pooled):
        return 0.0 if mt == mc else np.nan
    return (mt - mc) / pooled

def mean_by(df, group_cols, value_col):
    g = (
        df.dropna(subset=[value_col])
        .groupby(group_cols, observed=True)[value_col]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )
    se = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    return g.assign(ci_low=g["mean"] - 1.96 * se, ci_high=g["mean"] + 1.96 * se)

def weighted_mean_by(df, group_cols, value_col, weight_col="match_w"):
    """Weighted mean with approximate normal CI using effective n = sum(w)^2 / sum(w^2)."""
    rows = []
    for keys, g in df.dropna(subset=[value_col, weight_col]).groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        w = g[weight_col].astype(float).values
        x = g[value_col].astype(float).values
        wsum = w.sum()
        if wsum <= 0:
            continue
        mu = np.average(x, weights=w)
        # weighted variance
        var = np.average((x - mu) ** 2, weights=w)
        n_eff = (wsum ** 2) / np.sum(w ** 2)
        se = np.sqrt(var / max(n_eff, 1))
        row = dict(zip(group_cols, keys))
        row.update(mean=mu, sd=np.sqrt(var), n=len(g), n_eff=n_eff,
                   ci_low=mu - 1.96 * se, ci_high=mu + 1.96 * se)
        rows.append(row)
    return pd.DataFrame(rows)

def add_wilson_ci(df, count_col="k", n_col="n", z=1.96):
    n = df[n_col].astype(float)
    k = df[count_col].astype(float)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return df.assign(
        share=p,
        ci_low=(center - half).clip(lower=0),
        ci_high=(center + half).clip(upper=1),
    )

def assign_tp_bin(sa, intensity_col="tp_intensity", bin_col="tp_bin", p75=None):
    """Ordinal dose from a zero-inflated intensity.

    0 = no exposure; 1 = positive up to the 75th percentile of *non-zero*
    values; 2 = above that percentile (right-tail / high dose).
    """
    sa = sa.copy()
    x = pd.to_numeric(sa[intensity_col], errors="coerce").fillna(0.0).clip(lower=0)
    sa[intensity_col] = x
    if p75 is None:
        nz = x[x > 0]
        if nz.empty:
            raise ValueError(
                f"No non-zero {intensity_col} values to set the 75th percentile"
            )
        p75 = float(nz.quantile(0.75))
    p75 = float(p75)
    sa[bin_col] = np.where(x <= 0, 0, np.where(x <= p75, 1, 2)).astype(int)
    return sa, p75

def attach_intensities(sa):
    sa = sa.copy()
    sa["tp_intensity"] = intensity_per_1000(sa["n_detected"], sa["pop_total_2019"])
    sa["cand_intensity"] = intensity_per_1000(sa["n_candidates"], sa["pop_total_2019"])
    sa["atm_intensity"] = intensity_per_1000(sa["n_candidates_atm"], sa["pop_total_2019"])
    sa["comm_cand_intensity"] = intensity_per_1000(
        sa["n_candidates_comm"], sa["pop_total_2019"]
    )
    sa["noncomm_cand_intensity"] = intensity_per_1000(
        sa["n_candidates_noncomm"], sa["pop_total_2019"]
    )
    sa["atm_only_intensity"] = sa["atm_intensity"]  # alias: atm is flag-only
    sa, _ = assign_tp_bin(sa)
    return sa

def intensity_per_1000(n, pop):
    """Count per 1,000 residents (2019 population)."""
    return (
        n.astype(float) * 1000.0 / pop.astype(float)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def tp_intensity_per_1000(n_detected, pop):
    """Detected third places per 1,000 residents (2019 population)."""
    return intensity_per_1000(n_detected, pop)

def build_matching_panel(sa, win, covariates=None, treat_col=None):
    """Join SA treatment to pre/post match-window outcomes; attach LOVE_PRE_WINDOW pre and post shares."""
    covariates = list(_cfg("PS_COVARIATES") if covariates is None else covariates)
    treat_col = _cfg("TREAT_FLAG") if treat_col is None else treat_col
    pre_ts = _cfg("PRE_TS")
    match_post_ts = _cfg("MATCH_POST_TS")
    outcome_cols = _cfg("OUTCOME_COLS")
    share_outcome_cols = _cfg("SHARE_OUTCOME_COLS")
    match_win = _cfg("PSM_MATCH_WINDOW")
    love_win = _cfg("LOVE_PRE_WINDOW")
    out = win[win["win"] == match_win].drop(columns=["win"]).reset_index(drop=True)
    out["month"] = pd.to_datetime(out["month"])
    agg_cols = [c for c in outcome_cols if c in out.columns] + [
        c for c in ("n_workers", "n_worker_days") if c in out.columns
    ]
    pre_sa = (
        out[out["month"] == pre_ts]
        .set_index("yishuv_stat_2022")[agg_cols]
        .add_suffix("_pre")
    )
    post_sa = (
        out[out["month"] == match_post_ts]
        .set_index("yishuv_stat_2022")[agg_cols]
        .add_suffix("_post")
    )
    panel = (
        sa.set_index("yishuv_stat_2022")
        .join(pre_sa, how="inner")
        .join(post_sa, how="inner")
    )
    extra = win[win["win"] == love_win].copy()
    extra["month"] = pd.to_datetime(extra["month"])
    extra_cols = [c for c in share_outcome_cols if c in extra.columns]
    if extra_cols:
        extra_pre = (
            extra[extra["month"] == pre_ts]
            .set_index("yishuv_stat_2022")[extra_cols]
            .add_suffix(f"_{love_win}_pre")
        )
        extra_post = (
            extra[extra["month"] == match_post_ts]
            .set_index("yishuv_stat_2022")[extra_cols]
            .add_suffix(f"_{love_win}_post")
        )
        panel = panel.join(extra_pre, how="left").join(extra_post, how="left")
    panel = panel.reset_index()
    for c in outcome_cols:
        if f"{c}_post" in panel.columns and f"{c}_pre" in panel.columns:
            panel[f"d_{c}"] = panel[f"{c}_post"] - panel[f"{c}_pre"]
    for c in share_outcome_cols:
        pre_c, post_c = f"{c}_{love_win}_pre", f"{c}_{love_win}_post"
        if pre_c in panel.columns and post_c in panel.columns:
            panel[f"d_{c}_{love_win}"] = panel[post_c] - panel[pre_c]
    need = list(dict.fromkeys([*covariates, treat_col]))
    missing_treat = treat_col not in panel.columns
    if missing_treat:
        raise KeyError(f"{treat_col} missing on SA frame — set the binary flag before matching")
    return panel, panel.dropna(subset=need).copy(), out

def psm_match(panel_frame, covariates=None, caliper_sd=None, treat_col=None):
    """1:1 greedy NN on logit PS. Returns a dict; does not touch notebook globals."""
    covariates = list(_cfg("PS_COVARIATES") if covariates is None else covariates)
    caliper_sd = _cfg("CALIPER_SD") if caliper_sd is None else caliper_sd
    treat_col = _cfg("TREAT_FLAG") if treat_col is None else treat_col
    m = panel_frame.copy()
    X_raw = m[covariates].astype(float)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_raw), index=m.index, columns=covariates)
    y = m[treat_col].astype(int)
    ps_model = LogisticRegression(max_iter=5000, random_state=20260816)
    ps_model.fit(X, y)
    m["ps"] = ps_model.predict_proba(X)[:, 1]
    m["logit_ps"] = np.log(
        m["ps"].clip(1e-6, 1 - 1e-6) / (1.0 - m["ps"].clip(1e-6, 1 - 1e-6))
    )
    caliper = caliper_sd * float(m["logit_ps"].std(ddof=1))
    treated_idx = m.index[m[treat_col] == 1].tolist()
    control_idx = m.index[m[treat_col] == 0].tolist()
    logit = m["logit_ps"]
    ctrl_available = sorted(control_idx, key=lambda i: (float(logit.loc[i]), i))
    treated_order = sorted(treated_idx, key=lambda i: (-logit.loc[i], i))
    pairs = []
    for ti in treated_order:
        if not ctrl_available:
            break
        t_lp = float(logit.loc[ti])
        best_ci, best_d = None, None
        for ci in ctrl_available:
            d = abs(t_lp - float(logit.loc[ci]))
            if d <= caliper and (best_d is None or d < best_d):
                best_ci, best_d = ci, d
        if best_ci is not None:
            pairs.append((ti, best_ci, best_d))
            ctrl_available.remove(best_ci)
    matched_ids = sorted({i for pair in pairs for i in pair[:2]})
    matched = m.loc[matched_ids].copy()
    matched["match_w"] = 1.0
    matched["cem_w"] = 1.0
    m["matched"] = m.index.isin(matched_ids)
    pair_of = {}
    for k, (ti, ci, _d) in enumerate(pairs):
        pair_of[m.loc[ti, "yishuv_stat_2022"]] = k
        pair_of[m.loc[ci, "yishuv_stat_2022"]] = k
    matched["pair_id"] = matched["yishuv_stat_2022"].map(pair_of)
    pair_df = pd.DataFrame(pairs, columns=["treated_idx", "control_idx", "abs_logit_diff"])
    return {
        "m": m, "matched": matched, "pairs": pairs, "pair_df": pair_df,
        "ps_model": ps_model, "caliper": caliper, "covariates": covariates,
    }

def _smd_table(var_list, before_df, after_df, treat_col=None):
    treat_col = _cfg("TREAT_FLAG") if treat_col is None else treat_col
    rows = []
    for col, label in var_list:
        if col not in before_df.columns or col not in after_df.columns:
            continue
        t0 = before_df.loc[before_df[treat_col] == 1, col].astype(float)
        c0 = before_df.loc[before_df[treat_col] == 0, col].astype(float)
        t1 = after_df.loc[after_df[treat_col] == 1, col].astype(float)
        c1 = after_df.loc[after_df[treat_col] == 0, col].astype(float)
        rows.append({
            "col": col,
            "variable": label,
            "smd_before": standardised_mean_diff(t0, c0),
            "smd_after": standardised_mean_diff(t1, c1),
        })
    return pd.DataFrame(rows)

def _love_panel(ax, bal, title):
    y = np.arange(len(bal))
    ax.scatter(bal["smd_before"], y, marker="s", s=50, color="#a0aec0",
               label="Before PSM", zorder=3)
    ax.scatter(bal["smd_after"], y, marker="o", s=50, color="#2b6cb0",
               label="After PSM", zorder=3)
    for i, r in bal.iterrows():
        ax.plot([r["smd_before"], r["smd_after"]], [i, i],
                color="#cbd5e0", lw=1.2, zorder=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.1, color="#c53030", ls="--", lw=0.8, label="|SMD|=0.1")
    ax.axvline(-0.1, color="#c53030", ls="--", lw=0.8)
    ax.axvline(0.2, color="#c53030", ls=":", lw=0.8, label="|SMD|=0.2")
    ax.axvline(-0.2, color="#c53030", ls=":", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(bal["variable"])
    ax.set_xlabel("Standardised mean difference (treated − control)")
    ax.set_title(title)
    ax.set_ylim(len(bal) - 0.5, -0.5)
    ax.legend(loc="best", fontsize=8)

class AbsorbedOLS:
    """Results of a within regression; mirrors the statsmodels API used by `_row`."""

    def __init__(self, params, cov, names, nobs, df, n_clusters, k_total, dropped):
        self.params = pd.Series(params, index=names)
        self.cov = pd.DataFrame(cov, index=names, columns=names)
        var = np.diag(np.asarray(cov, dtype=float))
        self.bse = pd.Series(np.sqrt(np.where(var > 0, var, np.nan)), index=names)
        self.tvalues = self.params / self.bse
        self.pvalues = pd.Series(
            2 * stats.t.sf(np.abs(self.tvalues.to_numpy(dtype=float)), df),
            index=names,
        )
        self.nobs, self.df, self.n_clusters = nobs, df, n_clusters
        self.k_total, self.dropped = k_total, dropped

    def conf_int(self, alpha=0.05):
        crit = stats.t.ppf(1 - alpha / 2, self.df)
        return pd.DataFrame({0: self.params - crit * self.bse,
                             1: self.params + crit * self.bse})

    def wald_equal(self, terms):
        """F test that the listed coefficients are all equal."""
        terms = [t for t in terms if np.isfinite(self.bse.get(t, np.nan))]
        if len(terms) < 2:
            return np.nan, np.nan
        pos = {n: i for i, n in enumerate(self.params.index)}
        R = np.zeros((len(terms) - 1, len(self.params)))
        for j, t in enumerate(terms[1:]):
            R[j, pos[terms[0]]], R[j, pos[t]] = 1.0, -1.0
        Rb = R @ np.nan_to_num(self.params.to_numpy(dtype=float))
        RVR = R @ np.nan_to_num(self.cov.to_numpy(dtype=float)) @ R.T
        q = R.shape[0]
        F = float(Rb @ np.linalg.pinv(RVR) @ Rb / q)
        return F, float(stats.f.sf(F, q, self.df))

def _fit_absorb(data, y, x_terms, absorb, cluster_col, weights_col=None, tol=1e-8):
    """Cluster-robust OLS with the fixed effects absorbed instead of dummy-coded.

    Dummy-coding several hundred SA (or SA x window) effects and then asking for a
    cluster-robust covariance with 33 localities is numerically unsafe: the design is
    rank deficient, `pinv` hides that, and the reported standard errors moved by
    orders of magnitude between runs of the same specification. Absorbing the effects
    leaves a design of only `len(x_terms)` columns, so the covariance is well
    conditioned, and a regressor left without within variation is returned as
    unidentified (NaN) rather than given an arbitrary standard error.
    """
    need = [y, *x_terms, cluster_col, *absorb] + ([weights_col] if weights_col else [])
    d = data.dropna(subset=need)
    if d.empty:
        return None
    w = d[weights_col].astype(float).to_numpy() if weights_col else np.ones(len(d))
    groups = []
    for a in absorb:
        codes, uniq = pd.factorize(d[a])
        groups.append((codes, len(uniq)))
    raw = d[[y, *x_terms]].astype(float).to_numpy()
    Z = _demean(raw, groups, w)
    sw = np.sqrt(w)
    yt, Xt = Z[:, 0] * sw, Z[:, 1:] * sw[:, None]
    raw_scale = np.sqrt((raw[:, 1:] ** 2).mean(axis=0))
    keep = []
    for j in range(Xt.shape[1]):
        if np.sqrt((Xt[:, j] ** 2).mean()) <= tol * max(raw_scale[j], 1.0):
            continue  # fully absorbed by the fixed effects
        if np.linalg.matrix_rank(Xt[:, keep + [j]], tol=1e-9) == len(keep) + 1:
            keep.append(j)
    if not keep:
        return None
    dropped = [t for j, t in enumerate(x_terms) if j not in keep]
    Xk = Xt[:, keep]
    b = np.linalg.lstsq(Xk, yt, rcond=None)[0]
    u = yt - Xk @ b
    g, uniq_g = pd.factorize(d[cluster_col])
    n_g = len(uniq_g)
    scores = np.zeros((n_g, len(keep)))
    np.add.at(scores, g, Xk * u[:, None])
    XtX_inv = np.linalg.pinv(Xk.T @ Xk)
    n = len(d)
    # Absorbed effects still cost degrees of freedom; the shared constant is counted once.
    k_total = len(keep) + sum(nc for _, nc in groups) - (len(groups) - 1)
    corr = (n_g / (n_g - 1)) * ((n - 1) / max(n - k_total, 1))
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * corr
    params = np.full(len(x_terms), np.nan)
    cov = np.full((len(x_terms), len(x_terms)), np.nan)
    for a_i, ja in enumerate(keep):
        params[ja] = b[a_i]
        for c_i, jc in enumerate(keep):
            cov[ja, jc] = V[a_i, c_i]
    return AbsorbedOLS(params, cov, list(x_terms), n, n_g - 1, n_g, k_total, dropped)

def _row(res, term, extra=None):
    ci = res.conf_int().loc[term]
    out = {
        "coef": res.params[term], "se": res.bse[term], "t": res.tvalues[term],
        "p": res.pvalues[term], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1],
        "n_obs": int(res.nobs),
    }
    return {**out, **(extra or {})}

def fit_did_table(data, outcomes, wins=None, cluster_col="semel_yishuv",
                  weights_col=None, intensity_col=None, intensity2_col=None,
                  bin_col=None):
    """TWFE DiD for each outcome x window. `outcomes` is [(col, label), ...].

    Pass intensity2_col to estimate two intensity × post terms in the same TWFE.
    Pass bin_col to estimate did_bin1 and did_bin2 jointly (categorical dose).
    """
    intensity_col = _cfg("TREAT_FLAG") if intensity_col is None else intensity_col
    if wins is None:
        wins = _did_window_names(data["win"]) or sorted({str(w) for w in data["win"]})
    rows = []
    for w in wins:
        d = data[data["win"] == w]
        for col, lab in outcomes:
            if col not in d.columns:
                continue
            r = did_twfe(d, col, cluster_col=cluster_col, weights_col=weights_col,
                         intensity_col=intensity_col, intensity2_col=intensity2_col,
                         bin_col=bin_col)
            if not r:
                continue
            batch = r if isinstance(r, list) else [r]
            for row in batch:
                rows.append({**row, "win": w, "label": lab})
    out = pd.DataFrame(rows)
    if len(out):
        out["sig"] = out["p"].map(stars)
    return out

def build_long(sa_frame, months=None, ref=None, win_df=None, treat_col=None):
    """SA x month x window panel with treatment, post and fixed-effect keys.

    An SA x window cell is kept if it is observed in the reference month and in at
    least one other month. With pre + one matching post that is a balanced-panel
    requirement; extra post months (Jan 2023 next to Feb 2023) make it unbalanced.
    """
    treat_col = _cfg("TREAT_FLAG") if treat_col is None else treat_col
    pre_ts = _cfg("PRE_TS")
    months = [pre_ts, *_cfg("POST_TS")] if months is None else months
    ref = months[0] if ref is None else ref
    win_df = _cfg("df_win") if win_df is None else win_df
    keys = list(dict.fromkeys(k for k in _cfg("SA_KEYS") if k in sa_frame.columns))
    if treat_col not in keys and treat_col in sa_frame.columns:
        keys.insert(1, treat_col)
    for _extra in sa_frame.columns:
        if _extra not in keys and _extra in _INTENSITY_KEYS:
            keys.append(_extra)
    out = win_df[win_df["month"].isin(months)].merge(
        sa_frame[keys].drop_duplicates("yishuv_stat_2022"),
        on="yishuv_stat_2022",
        how="inner",
    )
    if treat_col not in out.columns:
        raise KeyError(f"{treat_col} missing on long panel — put the flag on sa_frame")
    out["treated"] = out[treat_col].astype(int)
    for _int_col in ("tp_intensity", "cand_intensity", "atm_intensity",
                     "atm_only_intensity", "comm_cand_intensity",
                     "noncomm_cand_intensity", "share_tp", "nontp_candidate_intensity"):
        if _int_col in out.columns:
            out[_int_col] = out[_int_col].astype(float).fillna(0.0)
    if "tp_bin" in out.columns:
        out["tp_bin"] = out["tp_bin"].astype(int)
    if "ipw_weight" in out.columns:
        out["ipw_weight"] = out["ipw_weight"].astype(float)
    out["post"] = (out["month"] > pre_ts).astype(int)
    out["month_str"] = out["month"].dt.strftime("%Y_%m")
    cover = (
        out.assign(is_ref=(out["month"] == ref).astype(int))
        .groupby(["yishuv_stat_2022", "win"])
        .agg(has_ref=("is_ref", "max"), n_months=("month", "nunique"))
        .reset_index()
    )
    keep = cover[(cover["has_ref"] == 1) & (cover["n_months"] >= 2)]
    out = out.merge(keep[["yishuv_stat_2022", "win"]], on=["yishuv_stat_2022", "win"])
    out["unit_win"] = out["yishuv_stat_2022"] + "_" + out["win"]
    out["month_win"] = out["month_str"] + "_" + out["win"]
    return out

def did_twfe(data, y, cluster_col="semel_yishuv", weights_col=None,
             unit="yishuv_stat_2022", time="month_str",
             intensity_col=None, intensity2_col=None, bin_col=None):
    """y_it = a_i + g_t + sum_k delta_k * (dose_k_i x post_t) + e_it.

    Default treatment is TREAT_FLAG (same binary as PSM). Pass tp_intensity /
    cand_intensity / atm_intensity (and optionally intensity2_col) for dosage.
    Pass bin_col='tp_bin' for two dummy interactions:
    did_bin1 = 1{bin=1} x post and did_bin2 = 1{bin=2} x post (WLS if weights_col).
    Coefficient = change in y per unit of intensity, or vs bin 0 for categorical dose.
    """
    if bin_col:
        need = [y, bin_col, "post", unit, time, cluster_col]
        if weights_col:
            need.append(weights_col)
        d = data.dropna(subset=need).copy()
        if d.empty or d["post"].nunique() < 2:
            return None
        d["did_bin1"] = ((d[bin_col] == 1).astype(float) * d["post"].astype(float))
        d["did_bin2"] = ((d[bin_col] == 2).astype(float) * d["post"].astype(float))
        terms = ["did_bin1", "did_bin2"]
        if any(d[t].nunique() < 2 for t in terms):
            return None
        res = _fit_absorb(d, y, terms, [unit, time], cluster_col, weights_col)
        if res is None:
            return None
        F, p_wald = res.wald_equal(terms)
        rows = []
        for name, k in zip(terms, (1, 2)):
            base = d.loc[(d[bin_col] == k) & (d["post"] == 0), y].mean()
            rows.append(_row(res, name, {
                "outcome": y,
                "intensity": f"{bin_col}={k}",
                "bin": k,
                "pre_mean_treated": base,
                "pct_of_pre": 100 * res.params[name] / base if base else np.nan,
                "n_sa": d[unit].nunique(),
                "n_clusters": d[cluster_col].nunique(),
                "wald_eq_F": F,
                "wald_eq_p": p_wald,
            }))
        return rows

    intensity_col = _cfg("TREAT_FLAG") if intensity_col is None else intensity_col
    ints = [intensity_col]
    if intensity2_col:
        if intensity2_col == intensity_col:
            raise ValueError("intensity2_col must differ from intensity_col")
        ints.append(intensity2_col)
    need = [y, *ints, "post", unit, time, cluster_col]
    d = data.dropna(subset=need).copy()
    if d.empty or d["post"].nunique() < 2:
        return None
    if any(d[c].nunique() < 2 for c in ints):
        return None
    terms = []
    for i, c in enumerate(ints, start=1):
        name = "did" if i == 1 else f"did{i}"
        d[name] = d[c].astype(float) * d["post"]
        terms.append(name)
    res = _fit_absorb(d, y, terms, [unit, time], cluster_col, weights_col)
    if res is None:
        return None
    rows = []
    for name, c in zip(terms, ints):
        base = d.loc[(d[c] > 0) & (d["post"] == 0), y].mean()
        rows.append(_row(res, name, {
            "outcome": y,
            "intensity": c,
            "pre_mean_treated": base,
            "pct_of_pre": 100 * res.params[name] / base if base else np.nan,
            "n_sa": d[unit].nunique(),
            "n_clusters": d[cluster_col].nunique(),
        }))
    return rows if intensity2_col else rows[0]

def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

def _demean(Z, groups, w, tol=1e-10, maxiter=500):
    """Sweep out weighted fixed effects by alternating projections."""
    Z = np.array(Z, dtype=float, copy=True)
    for _ in range(maxiter):
        step_max = 0.0
        for codes, ncat in groups:
            num = np.zeros((ncat, Z.shape[1]))
            den = np.zeros(ncat)
            np.add.at(num, codes, Z * w[:, None])
            np.add.at(den, codes, w)
            step = (num / np.where(den == 0, 1.0, den)[:, None])[codes]
            Z -= step
            step_max = max(step_max, float(np.abs(step).max()))
        if len(groups) < 2 or step_max < tol:
            break
    return Z

def run_psm_did(treat=None, *, flag_col=None, sa=None, win=None,
                plot=True, verbose=True, set_globals=True,
                ps_balance_vars=None, pre_share_vars=None):
    """1:1 PSM on treat_rule (or `treat` callable), then TWFE DiD of treat × post.

    Parameters
    ----------
    treat : callable, optional
        ``treat(sa) -> 0/1``. Default ``treat_rule`` from the config cell.
        Example: ``run_psm_did(lambda sa: sa["n_candidates_atm"] >= 1)``.
    flag_col : str, optional
        Name of the 0/1 column (default ``TREAT_FLAG``).
    sa, win : DataFrame, optional
        Defaults ``df_sa`` / ``df_win``. Work is on copies; ``df_sa`` is
        updated only if ``set_globals``.
    plot : bool
        Fig. 1 (love plot), PS overlap, and Fig. 2 (DiD coefficients).
    verbose : bool
        Print matching stats, Table 1, and paper captions.
    set_globals : bool
        Write TREAT_FLAG, matched, long_matched, did, … so later cells keep working.
    ps_balance_vars : list of (col, label), optional
        Love-plot Panel A. If omitted, uses the notebook's ``ps_balance_vars``.
    pre_share_vars : list of (col, label), optional
        Love-plot Panel B. If omitted, uses the notebook's ``pre_share_vars``.

    Returns
    -------
    PsmDidResult
    """
    flag_col = _cfg("TREAT_FLAG") if flag_col is None else flag_col
    rule = _cfg("treat_rule") if treat is None else treat
    sa = apply_treat_flag(_cfg("df_sa") if sa is None else sa, flag_col=flag_col, rule=rule)
    win = _cfg("df_win") if win is None else win
    covariates = list(_cfg("PS_COVARIATES"))
    caliper_sd = _cfg("CALIPER_SD")
    outcomes = list(_cfg("MAIN_OUTCOMES"))
    treat_label = _cfg("TREAT_LABEL")
    rule_name = getattr(rule, "__name__", "treat")

    if verbose:
        print(f"Treat = {rule_name}  →  {flag_col}")
        try:
            print(inspect.getsource(rule).strip())
        except (OSError, TypeError):
            print(repr(rule))
        print(sa[flag_col].value_counts().rename("n_SA"))
        print(f"P(treated) = {sa[flag_col].mean():.3f}")

    panel, panel_m, df_out = build_matching_panel(
        sa, win, covariates=covariates, treat_col=flag_col,
    )
    if verbose:
        print(f"Panel SAs with pre + post: {len(panel):,}")
        print(f"Matching sample (non-null covariates): {len(panel_m):,}")
        print(panel_m[flag_col].value_counts().rename("n_SA"))

    ps = psm_match(panel_m, covariates=covariates, caliper_sd=caliper_sd, treat_col=flag_col)
    m, matched, pairs, pair_df = ps["m"], ps["matched"], ps["pairs"], ps["pair_df"]
    ps_model, caliper = ps["ps_model"], ps["caliper"]
    matched = matched.copy()
    matched["treat_label"] = np.where(matched[flag_col] == 1, treat_label, "Control")
    treated = matched.loc[matched[flag_col] == 1].copy()
    control = matched.loc[matched[flag_col] == 0].copy()
    n_t0 = int((m[flag_col] == 1).sum())
    n_c0 = int((m[flag_col] == 0).sum())
    n_t1 = int(len(treated))
    n_c1 = int(len(control))
    if verbose:
        print("PSM covariates (z-scored):", covariates)
        print(f"PS caliper: {caliper_sd} × SD(logit PS) = {caliper:.4f}")
        print(f"PS grouping: {flag_col} from {rule_name}")
        print(f"Before PSM: treated={n_t0} control={n_c0}")
        print(f"After PSM:  treated={n_t1} control={n_c1}")
        print(f"Pairs: {len(pairs):,}  |  treated dropped: {n_t0 - n_t1:,}")
        if len(pair_df):
            print(f"Mean |Δ logit PS|: {pair_df['abs_logit_diff'].mean():.4f}")
        print("Logit coefficients on z-scored covariates (log-odds per +1 SD):")
        print(pd.Series(ps_model.coef_.ravel(), index=covariates))

    long_matched = build_long(matched, treat_col=flag_col)
    long_full = build_long(m, treat_col=flag_col)
    if verbose:
        print(f"long_matched: {len(long_matched):,} rows, "
              f"{long_matched['yishuv_stat_2022'].nunique():,} SAs")
        print(f"long_full:    {len(long_full):,} rows, "
              f"{long_full['yishuv_stat_2022'].nunique():,} SAs")
        print(f"pairs mapped: {int(matched['pair_id'].notna().sum()):,} of {len(matched):,} SAs")
        print("Balanced SA count by window (matched sample):")
        print(long_matched.groupby("win")["yishuv_stat_2022"].nunique().rename("n_SA"))

    did_wins = _did_window_names(long_matched["win"])
    did = fit_did_table(long_matched, outcomes, wins=did_wins,
                        intensity_col=flag_col)
    _order = [lab for _, lab in outcomes if lab in set(did["label"])]
    _show = ["label", "coef", "se", "p", "sig", "pre_mean_treated", "pct_of_pre",
             "n_obs", "n_sa", "n_clusters"]
    primary_window = _cfg("PRIMARY_WINDOW")
    if verbose:
        print(f"\n=== Primary window: {primary_window} ===")
        print(did[did["win"] == primary_window][_show]
              .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        print("\n=== DiD coefficient by window (treat × post) ===")
        print(did.pivot(index="label", columns="win", values="coef")
                 .reindex(_order)[did_wins].round(4).to_string())
        print("\n=== p-values ===")
        print(did.pivot(index="label", columns="win", values="p")
                 .reindex(_order)[did_wins].round(3).to_string())
        n_out_exp = len(outcomes)
        print(f"\nSpecs: {len(did)}  (expected {n_out_exp * len(did_wins)} = "
              f"{n_out_exp} outcomes × {len(did_wins)} windows)")
        print(
            f"\n**Table 1.** Two-way fixed-effects difference-in-differences: "
            f"`{flag_col}` × post on weekday local-share outcomes, by time-of-day "
            f"window. *Notes:* Unit = home statistical area × month; SA and month "
            f"fixed effects; standard errors clustered on locality; "
            f"n = {n_t1} matched pairs; pre = {_cfg('PRE_LABEL')}, post = {_cfg('POST_LABEL')} "
            f"(matched on {_cfg('MATCH_POST_LABEL')}). Treatment is the same binary as PSM "
            f"(`{rule_name}`). Home radius = {_cfg('OUTBOUND_M')} m; "
            f"near-work cut = {_cfg('WORK_NEAR_M')} m. Shares are fractions, so 0.01 = 1 pp. "
            f"Windows: {', '.join(did_wins)}."
        )

    if plot:
        bal_x_vars = list(
            ps_balance_vars if ps_balance_vars is not None else _cfg("ps_balance_vars")
        )
        bal_y_vars = list(
            pre_share_vars if pre_share_vars is not None else _cfg("pre_share_vars")
        )
        if not bal_x_vars:
            raise ValueError("ps_balance_vars is empty — pass a list of (col, label)")
        if not bal_y_vars:
            raise ValueError("pre_share_vars is empty — pass a list of (col, label)")
        bal_x = _smd_table(bal_x_vars, m, matched, treat_col=flag_col)
        bal_y = _smd_table(bal_y_vars, m, matched, treat_col=flag_col)
        if verbose:
            print("\nPanel A — PS covariates and intensities")
            print(bal_x.to_string(index=False))
            print(f"\nPanel B — {len(bal_y)} pre-period outcome shares")
            print(bal_y.to_string(index=False))
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(14, 8), sharex=True,
            gridspec_kw={"width_ratios": [1, 1.35]},
        )
        _love_panel(ax0, bal_x, "PS covariates and treatment intensity")
        _love_panel(ax1, bal_y, f"Pre-period local shares ({len(bal_y)} rows)")
        fig.suptitle(
            f"Covariate and outcome-share balance before and after PSM ({flag_col})",
            y=1.01,
        )
        plt.tight_layout()
        plt.show()
        if verbose:
            print(
                f"**Fig. 1.** Standardised mean differences (treated minus control) "
                f"before and after 1:1 PSM on z-scored covariates (logit PS caliper "
                f"{caliper_sd} SD). Left: PS covariates ({', '.join(covariates)}) "
                f"and third-place intensities (not in the PS). Right: Jan 2020 "
                f"local-share outcomes ({len(bal_y)} rows); only `share_local_poi_h_pre` is in the logit. "
                f"*Notes:* Unit = home statistical area; n = {n_t1} matched pairs; "
                f"treatment = `{rule_name}`; dashed lines at |SMD| = 0.1, "
                f"dotted at |SMD| = 0.2. Intensities should remain different after matching."
            )
            _ps_bal = bal_x.loc[~bal_x["variable"].str.contains("not in PS", regex=False)]
            if len(_ps_bal):
                _worst = _ps_bal.loc[_ps_bal["smd_after"].abs().idxmax()]
                print(f"Largest residual imbalance among PS covariates: "
                      f"{_worst['variable']} (SMD {_worst['smd_after']:+.3f})")
            _off = bal_y.loc[bal_y["smd_after"].abs() > 0.1]
            if _off.empty:
                print(f"All {len(bal_y)} pre-period outcome shares have |SMD| ≤ 0.1 after matching.")
            else:
                print(f"{len(_off)} of {len(bal_y)} pre-period outcome shares have |SMD| > 0.1 "
                      f"after matching:")
                print(_off[["variable", "smd_before", "smd_after"]].to_string(index=False))

        fig, ax = plt.subplots(figsize=(6, 3.5))
        for lab, flag, color in [("Control", 0, "#a0aec0"), ("Treated", 1, "#2b6cb0")]:
            ax.hist(
                m.loc[m[flag_col] == flag, "logit_ps"],
                bins=25, alpha=0.55, label=f"{lab} (all)", color=color, density=True,
            )
        ax.set_xlabel("Logit propensity score")
        ax.set_ylabel("Density")
        ax.set_title(f"Propensity-score overlap (full sample, {flag_col})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        if verbose and len(pair_df):
            print(
                f"Matched pair |Δ logit PS|: median={pair_df['abs_logit_diff'].median():.4f}, "
                f"p95={pair_df['abs_logit_diff'].quantile(0.95):.4f}, "
                f"max={pair_df['abs_logit_diff'].max():.4f}"
            )

        win_plot = [w for w in did_wins if w in set(did["win"])]
        n_out = len(outcomes)
        n_col = 4
        n_row = int(np.ceil(n_out / n_col))
        fig, axes = plt.subplots(n_row, n_col, figsize=(12.5, 3.1 * n_row), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (col, title) in zip(axes, outcomes):
            sub = did[did["outcome"] == col].set_index("win").reindex(win_plot)
            yv = np.arange(len(win_plot))
            colours = ["#2b6cb0" if p < 0.10 else "#a0aec0" for p in sub["p"].fillna(1)]
            ax.errorbar(
                sub["coef"], yv,
                xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                fmt="none", ecolor="#2b6cb0", elinewidth=1.3, capsize=3,
                label="95% CI",
            )
            ax.scatter(sub["coef"], yv, s=40, c=colours, zorder=3)
            ax.axvline(0, ls="--", color="#dd6b20", lw=1.1)
            ax.set_yticks(yv)
            ax.set_yticklabels(win_plot)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("DiD coefficient (share points)")
            ax.tick_params(labelsize=8)
            ax.invert_yaxis()
        for ax in axes[n_out:]:
            ax.set_visible(False)
        axes[0].set_ylabel("Time-of-day window")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2b6cb0",
                       markersize=7, label="p < 0.10"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#a0aec0",
                       markersize=7, label="p ≥ 0.10"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
        plt.suptitle(f"{flag_col} × post by window, matched sample", y=1.02)
        plt.tight_layout()
        plt.show()
        if verbose:
            print(
                f"**Fig. 2.** `{flag_col}` × post coefficients with 95% cluster-robust "
                f"confidence intervals, by time-of-day window. Filled blue markers are "
                f"significant at 10%; grey markers are not. *Notes:* Same specification "
                f"as Table 1 (SA and month fixed effects, locality-clustered standard "
                f"errors). Windows: {', '.join(win_plot)}. Shares are fractions of the "
                f"denominator (named POIs / not-home-work places / commercial POIs), "
                f"not of all time. Treatment = `{rule_name}`."
            )

    out = PsmDidResult(
        flag_col=flag_col,
        panel=panel, panel_m=panel_m, df_out=df_out,
        m=m, matched=matched, treated=treated, control=control,
        pairs=pairs, pair_df=pair_df,
        ps=ps, ps_model=ps_model, caliper=caliper,
        long_matched=long_matched, long_full=long_full,
        did=did, did_wins=did_wins,
    )
    if set_globals:
        _write(
            TREAT_FLAG=flag_col,
            df_sa=sa,
            panel=panel,
            panel_m=panel_m,
            df_out=df_out,
            m=m,
            matched=matched,
            treated=treated,
            control=control,
            pairs=pairs,
            pair_df=pair_df,
            ps=ps,
            ps_model=ps_model,
            caliper=caliper,
            long_matched=long_matched,
            long_full=long_full,
            did=did,
            DID_WINS=did_wins,
            MAIN_OUTCOMES=outcomes,
            OUT_LABELS=dict(outcomes),
        )
    return out

class PsmDidResult(SimpleNamespace):
    """Short repr so Jupyter does not dump the matched frames."""

    def __repr__(self):
        n_pairs = 0
        matched = getattr(self, "matched", None)
        flag = getattr(self, "flag_col", None)
        if matched is not None and flag in getattr(matched, "columns", []):
            n_pairs = int((matched[flag] == 1).sum())
        n_did = 0 if getattr(self, "did", None) is None else len(self.did)
        return (
            f"PsmDidResult(flag={self.flag_col!r}, "
            f"pairs={n_pairs:,}, treated={n_pairs:,}, control={n_pairs:,}, "
            f"did_rows={n_did})"
        )

def run_cbgps_did(intensity_col, *, panel=None, plot=True, verbose=True,
                  set_globals=True, extra_gps_x=None, intensity2_col=None):
    """Fit CBGPS on log1p(intensity_col), then TWFE DiD of intensity × post.

    Parameters
    ----------
    intensity_col : str
        One of tp_intensity / cand_intensity / atm_intensity / share_tp /
        atm_only_intensity / comm_cand_intensity / noncomm_cand_intensity.
        share_tp is built as tp_intensity / cand_intensity if missing.
    panel : DataFrame, optional
        SA-level frame (default ``panel_m``).
    plot : bool
        Draw Fig. 2b (balance) and Fig. 2c (DiD coefficients).
    verbose : bool
        Print sample, weights, tables, and paper captions.
    extra_gps_x : str or list of str, optional
        Extra GPS covariates appended to ``PS_COVARIATES``. Example:
        ``extra_gps_x="nontp_candidate_intensity"`` adds
        ``cand_intensity - tp_intensity`` (non-detected candidate stock).
        The treatment column is never included in X.
    intensity2_col : str, optional
        Second DiD interaction, estimated jointly with ``intensity_col``:
        y = a_i + g_t + d1 (intensity2 x post) + d2 (intensity x post) + e.
        Typical: ``intensity2_col="nontp_candidate_intensity"``.
    set_globals : bool
        Write ``INTENSITY_COL``, ``gps``, ``fit_gps``, ``long_gps``, ``did_gps``
        so placebo / event study / Part 7 keep working.

    Returns
    -------
    SimpleNamespace with intensity_col, gps, fit_gps, long_gps, did, bal_plot.
    """
    if CBPS is None:
        raise ImportError("run_cbgps_did requires the cbps package")
    if intensity_col not in _ALLOWED_INT:
        raise ValueError(f"intensity_col must be one of {_ALLOWED_INT}")
    panel = (_cfg("panel_m") if panel is None else panel).copy()
    gps_covariates = list(_cfg("PS_COVARIATES"))
    outcomes = list(_cfg("MAIN_OUTCOMES"))

    # share_tp = detected / candidate intensity (same as share_detected).
    # Built here so run_cbgps_did("share_tp") does not need a prior assignment.
    if intensity_col == "share_tp" and "share_tp" not in panel.columns:
        if "share_detected" in panel.columns:
            panel["share_tp"] = panel["share_detected"].astype(float)
        elif {"tp_intensity", "cand_intensity"}.issubset(panel.columns):
            cand = panel["cand_intensity"].astype(float)
            panel["share_tp"] = np.where(
                cand > 0, panel["tp_intensity"].astype(float) / cand, 0.0
            )
        else:
            raise KeyError(
                "Cannot build share_tp: need share_detected or "
                "tp_intensity and cand_intensity on the panel"
            )
        panel["share_tp"] = (
            pd.to_numeric(panel["share_tp"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0, upper=1)
        )
        if verbose:
            print("Built share_tp = tp_intensity / cand_intensity "
                  "(0 if no candidates)")

    # Non-TP candidate stock: candidates that are not detected third places.
    if {"tp_intensity", "cand_intensity"}.issubset(panel.columns):
        panel["nontp_candidate_intensity"] = (
            panel["cand_intensity"].astype(float)
            - panel["tp_intensity"].astype(float)
        ).clip(lower=0)

    extra = [] if extra_gps_x is None else (
        [extra_gps_x] if isinstance(extra_gps_x, str) else list(extra_gps_x)
    )
    gps_x = list(dict.fromkeys([*gps_covariates, *extra]))
    if intensity_col in gps_x:
        gps_x = [c for c in gps_x if c != intensity_col]
        if verbose:
            print(f"Dropped {intensity_col} from GPS X (treatment cannot be a covariate)")

    need = [intensity_col, *gps_x, "yishuv_stat_2022", "semel_yishuv"]
    missing = [c for c in need if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing on panel: {missing} — re-run Parts 2–4")

    gps = panel.dropna(subset=[intensity_col, *gps_x]).copy()
    gps[intensity_col] = gps[intensity_col].astype(float).fillna(0.0).clip(lower=0)
    gps["t_gps"] = np.log1p(gps[intensity_col])
    # Z-score GPS covariates (mean 0, sd 1) so CBPS is not dominated by scale.
    # Original columns on `gps` are left unchanged for later DiD / tables.
    gps_fit = gps.copy()
    for c in gps_x:
        x = gps_fit[c].astype(float)
        sd = float(x.std(ddof=0))
        gps_fit[c] = 0.0 if (not np.isfinite(sd) or sd == 0) else (x - x.mean()) / sd
    formula = "t_gps ~ " + " + ".join(gps_x)
    if verbose:
        print(f"CBGPS sample: {len(gps):,} SAs (covariate-complete panel)")
        print(f"Intensity: {intensity_col}  →  GPS on log1p({intensity_col})")
        print(f"X (z-scored): {gps_x}")
        print(f"Formula: {formula}")
        print(gps[intensity_col].describe().rename("intensity"))

    fit_gps = CBPS(formula=formula, data=gps_fit, att=0, method="over")
    gps["gps_w"] = np.asarray(fit_gps.weights, dtype=float)
    if verbose:
        print(fit_gps.summary())
        print(
            f"Weights: min={gps['gps_w'].min():.4f}  mean={gps['gps_w'].mean():.4f}  "
            f"max={gps['gps_w'].max():.4f}  "
            f"ESS={gps['gps_w'].sum() ** 2 / (gps['gps_w'] ** 2).sum():.1f}"
        )

    # --- balance (Fig. 2b) ---
    bal_gps = balance(fit_gps)
    _before_key = "unweighted" if "unweighted" in bal_gps else "original"
    before = pd.DataFrame(bal_gps[_before_key]).copy()
    after = pd.DataFrame(bal_gps["balanced"]).copy()
    corr_col = "corr" if "corr" in before.columns else before.columns[0]
    rows = []
    for col in gps_x:
        if col not in before.index:
            continue
        rows.append({
            "covariate": col,
            "label": GPS_X_LABELS.get(col, col),
            "corr_before": float(before.loc[col, corr_col]),
            "corr_after": float(after.loc[col, corr_col]),
        })
    bal_plot = pd.DataFrame(rows)
    bal_plot["abs_before"] = bal_plot["corr_before"].abs()
    bal_plot["abs_after"] = bal_plot["corr_after"].abs()
    if verbose:
        print("CBGPS |corr(X, T)| by covariate:")
        print(bal_plot[["label", "corr_before", "corr_after", "abs_before", "abs_after"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        _off = bal_plot.loc[bal_plot["abs_after"] > 0.1]
        if _off.empty:
            print("All GPS covariates have |corr| ≤ 0.1 after weighting.")
        else:
            print(f"{len(_off)} covariate(s) still have |corr| > 0.1 after weighting:")
            print(_off[["label", "corr_after"]].to_string(index=False))

    if plot:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        y = np.arange(len(bal_plot))
        ax.scatter(bal_plot["abs_before"], y, marker="s", s=55, color="#a0aec0",
                   label="Before CBGPS", zorder=3)
        ax.scatter(bal_plot["abs_after"], y, marker="o", s=55, color="#2b6cb0",
                   label="After CBGPS", zorder=3)
        for i, r in bal_plot.iterrows():
            ax.plot([r["abs_before"], r["abs_after"]], [i, i],
                    color="#cbd5e0", lw=1.2, zorder=1)
        ax.axvline(0.1, color="#c53030", ls="--", lw=0.8, label="|corr|=0.1")
        ax.axvline(0.2, color="#c53030", ls=":", lw=0.8, label="|corr|=0.2")
        ax.set_yticks(y)
        ax.set_yticklabels(bal_plot["label"])
        ax.set_xlabel(f"|Pearson corr| with log1p({intensity_col})")
        ax.set_ylabel("GPS covariate")
        ax.set_title(f"Covariate–treatment balance before and after CBGPS ({intensity_col})")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(len(bal_plot) - 0.5, -0.5)
        ax.set_xlim(left=0)
        plt.tight_layout()
        plt.show()
        if verbose:
            print(
                f"**Fig. 2b.** Absolute Pearson correlation of each GPS covariate with "
                f"log1p(`{intensity_col}`) before vs after CBGPS weighting. Grey squares = "
                f"unweighted; blue circles = weighted. *Notes:* Unit = home statistical area; "
                f"n = {len(gps):,} covariate-complete SAs (`panel_m`, not 1:1 matched pairs); "
                f"X = {', '.join(gps_x)} (z-scored). Dashed line at |corr| = 0.1. Correlations from "
                f"`cbps.balance`; weights from `cbps.CBPS` (method='over')."
            )

    # --- DiD (Table 1b / Fig. 2c) ---
    long_gps = build_long(gps)
    # Always attach this call's dosage + GPS weights from `gps`.
    # build_long only copies SA_KEYS, so extra intensities (share_tp) would
    # otherwise be missing even though they exist on the SA frame.
    _attach_cols = ["yishuv_stat_2022", "gps_w", intensity_col]
    if intensity2_col:
        _attach_cols.append(intensity2_col)
    _attach_cols = list(dict.fromkeys(_attach_cols))
    missing_g = [c for c in _attach_cols if c not in gps.columns]
    if missing_g:
        raise KeyError(f"Missing on gps (need for DiD): {missing_g}")
    _attach = gps[_attach_cols].drop_duplicates("yishuv_stat_2022")
    long_gps = long_gps.drop(
        columns=[c for c in _attach_cols if c != "yishuv_stat_2022" and c in long_gps.columns]
    )
    long_gps = long_gps.merge(_attach, on="yishuv_stat_2022", how="inner")
    _need_long = [intensity_col] + ([intensity2_col] if intensity2_col else [])
    _miss_long = [c for c in _need_long if c not in long_gps.columns]
    if _miss_long:
        raise KeyError(f"{_miss_long} missing on long_gps after merge from gps")
    did_wins = _did_window_names(long_gps["win"])
    if verbose:
        print(f"long_gps: {len(long_gps):,} rows, "
              f"{long_gps['yishuv_stat_2022'].nunique():,} SAs, "
              f"windows={did_wins}")

    did = fit_did_table(
        long_gps, outcomes,
        wins=did_wins, intensity_col=intensity_col,
        weights_col="gps_w", intensity2_col=intensity2_col,
    )
    if intensity2_col:
        got = set(did["intensity"].dropna().astype(str)) if "intensity" in did.columns else set()
        if intensity2_col not in got:
            raise RuntimeError(
                f"Second intensity {intensity2_col!r} missing from DiD "
                f"(got {sorted(got)}). Re-run this cell."
            )
    _order = [lab for _, lab in outcomes if lab in set(did["label"])]
    _show = ["label", "intensity", "coef", "se", "p", "sig",
             "pre_mean_treated", "pct_of_pre", "n_obs", "n_sa", "n_clusters"]
    _show = [c for c in _show if c in did.columns]
    _ints = [intensity_col] + ([intensity2_col] if intensity2_col else [])
    _int_lab = {
        "tp_intensity": "TP intensity",
        "nontp_candidate_intensity": "Non-TP candidate intensity",
        "cand_intensity": "Candidate intensity",
        "atm_intensity": "RW-atmosphere intensity",
        "share_tp": "Share detected",
        "atm_only_intensity": "Atmosphere-only intensity",
        "comm_cand_intensity": "Commercial candidate intensity",
        "noncomm_cand_intensity": "Non-commercial candidate intensity",
    }
    if verbose:
        _treat = " + ".join(f"`{c}` × post" for c in _ints)
        print(f"Treatment: {_treat}  |  weights: gps_w")
        for _win in did_wins:
            print(f"\n=== CBGPS-weighted DiD, window: {_win} ===")
            print(did[did["win"] == _win][_show]
                  .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        def _star_cell(r):
            mark = r["sig"] if isinstance(r.get("sig"), str) else stars(r["p"])
            if mark:
                return f"{r['coef']:.3f}{mark}"
            return "--"

        _star = did.copy()
        _star["_cell"] = [_star_cell(r) for _, r in _star.iterrows()]
        print("\n=== DiD (significant coef × post; else --) ===")
        if intensity2_col and "intensity" in _star.columns:
            for icol in _ints:
                sub = _star[_star["intensity"] == icol]
                if sub.empty:
                    print(f"\n  {icol} × post: (no rows)")
                    continue
                wide = (
                    sub.pivot_table(index="label", columns="win", values="_cell",
                                    aggfunc="first")
                    .reindex(index=_order)
                    .reindex(columns=did_wins)
                    .fillna("--")
                )
                print(f"\n  {icol} × post")
                print(wide.to_string())
        else:
            wide = (
                _star.pivot_table(index="label", columns="win", values="_cell",
                                  aggfunc="first")
                .reindex(index=_order)
                .reindex(columns=did_wins)
                .fillna("--")
            )
            print(wide.to_string())
        _n_exp = len(outcomes) * len(did_wins) * (2 if intensity2_col else 1)
        print(f"\nSpecs: {len(did)}  (expected {_n_exp} = "
              f"{len(outcomes)} outcomes × {len(did_wins)} windows"
              f"{' × 2 intensities' if intensity2_col else ''})")
        print(
            f"\n**Table 1b.** Two-way fixed-effects difference-in-differences: "
            f"{_treat} on weekday local-share outcomes, with CBGPS "
            f"inverse-probability weights. *Notes:* Unit = home statistical area × month; "
            f"SA and month fixed effects; standard errors clustered on locality; "
            f"n = {long_gps['yishuv_stat_2022'].nunique():,} GPS-weighted SAs "
            f"(covariate-complete `panel_m`, not 1:1 matched pairs); "
            f"pre = {_cfg('PRE_LABEL')}, post = {_cfg('POST_LABEL')}. "
            f"GPS fit on log1p({intensity_col}) with z-scored X = {', '.join(gps_x)}; DiD uses the "
            f"original intensity scale. Shares are fractions, so 0.01 = 1 pp. "
            f"Windows: {', '.join(did_wins)}."
        )

    if plot:
        win_plot = [w for w in did_wins if w in set(did["win"])]
        n_out = len(outcomes)
        n_col = 4
        n_row = int(np.ceil(n_out / n_col))
        fig, axes = plt.subplots(n_row, n_col, figsize=(12.5, 3.1 * n_row), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        _term_col = {
            intensity_col: "#2b6cb0",
            (intensity2_col or ""): "#c05621",
        }
        for ax, (col, title) in zip(axes, outcomes):
            yv = np.arange(len(win_plot))
            ax.axvline(0, ls="--", color="#dd6b20", lw=1.1)
            for k, icol in enumerate(_ints):
                sub = (
                    did[(did["outcome"] == col) & (did["intensity"] == icol)]
                    .set_index("win")
                    .reindex(win_plot)
                )
                off = 0.0 if len(_ints) == 1 else (k - (len(_ints) - 1) / 2) * 0.22
                colr = _term_col.get(icol, "#2b6cb0")
                ax.errorbar(
                    sub["coef"], yv + off,
                    xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                    fmt="none", ecolor=colr, elinewidth=1.3, capsize=3,
                )
                ax.scatter(sub["coef"], yv + off, s=40, c=colr, zorder=3)
            ax.set_yticks(yv)
            ax.set_yticklabels(win_plot)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("DiD coef (intensity × post)")
            ax.tick_params(labelsize=8)
            ax.invert_yaxis()
        for ax in axes[n_out:]:
            ax.set_visible(False)
        axes[0].set_ylabel("Time-of-day window")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_term_col.get(c, "#2b6cb0"),
                       markersize=7, label=_int_lab.get(c, c) + " × post")
            for c in _ints
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
        plt.suptitle(
            "CBGPS-weighted intensity × post by window",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()
        if verbose:
            _treat = " and ".join(f"`{c}` × post" for c in _ints)
            print(
                f"**Fig. 2c.** {_treat} coefficients with 95% cluster-robust "
                f"CIs, CBGPS-weighted. *Notes:* Same TWFE as Table 1b; "
                f"windows {', '.join(win_plot)}; sample is covariate-complete SAs with CBGPS "
                f"weights, not the binary 1:1 matched pairs from Fig. 2."
            )

    out = CbgpsDidResult(
        intensity_col=intensity_col,
        intensity2_col=intensity2_col,
        gps=gps,
        fit_gps=fit_gps,
        long_gps=long_gps,
        did=did,
        bal_plot=bal_plot,
        gps_x=gps_x,
        formula=formula,
    )
    if set_globals:
        _write(
            INTENSITY_COL=intensity_col,
            INTENSITY2_COL=intensity2_col,
            DID_WINS=did_wins,
            gps=gps,
            fit_gps=fit_gps,
            long_gps=long_gps,
            did_gps=did,
            bal_gps=bal_gps,
            bal_plot=bal_plot,
        )
    return out

class CbgpsDidResult(SimpleNamespace):
    """Same fields as before; short repr so Jupyter does not dump `gps`."""

    def __repr__(self):
        n_sa = 0
        if getattr(self, "gps", None) is not None and "yishuv_stat_2022" in self.gps.columns:
            n_sa = int(self.gps["yishuv_stat_2022"].nunique())
        n_did = 0 if getattr(self, "did", None) is None else len(self.did)
        return (
            f"CbgpsDidResult(intensity={self.intensity_col!r}, "
            f"intensity2={self.intensity2_col!r}, "
            f"n_sa={n_sa:,}, did_rows={n_did})"
        )

def run_mnipw_did(*, panel=None, intensity_col="tp_intensity", bin_col="tp_bin",
                  p75=None, plot=True, verbose=True, set_globals=True):
    """Multinomial logit IPW on tp_bin, then WLS TWFE of bin × post.

    ATT weights: W_i = 1 if T=0, else P(T=0|X) / P(T=k|X). X is z-scored
    PS_COVARIATES (never intensity). DiD terms are did_bin1 and did_bin2.
    """
    panel = (_cfg("panel_m") if panel is None else panel).copy()
    gps_x = list(_cfg("PS_COVARIATES"))
    outcomes = list(_cfg("MAIN_OUTCOMES"))
    if p75 is None:
        p75 = _cfg("TP_BIN_P75", default=None)
    if intensity_col not in panel.columns:
        raise KeyError(f"{intensity_col} missing on panel — re-run Parts 2–3")
    missing_x = [c for c in gps_x if c not in panel.columns]
    if missing_x:
        raise KeyError(f"Missing GPS covariates: {missing_x}")

    gps = panel.dropna(subset=[intensity_col, *gps_x, "yishuv_stat_2022"]).copy()
    gps, p75 = assign_tp_bin(gps, intensity_col=intensity_col, bin_col=bin_col, p75=p75)
    if gps[bin_col].nunique() < 3:
        raise ValueError(
            f"{bin_col} needs three levels (0/1/2); got {sorted(gps[bin_col].unique())}"
        )
    if verbose:
        print(f"MN-IPW sample: {len(gps):,} covariate-complete SAs")
        print(f"Intensity: {intensity_col}  →  {bin_col}  (p75 of non-zeros = {p75:.4f})")
        print(gps[bin_col].value_counts().sort_index().rename("n_SA"))
        print(gps.groupby(bin_col)[intensity_col].agg(["min", "median", "max"]))

    gps_fit = gps.copy()
    for c in gps_x:
        x = gps_fit[c].astype(float)
        sd = float(x.std(ddof=0))
        gps_fit[c] = 0.0 if (not np.isfinite(sd) or sd == 0) else (x - x.mean()) / sd
    y = gps[bin_col].astype(int)
    X = sm.add_constant(gps_fit[gps_x].astype(float), has_constant="add")
    mn = sm.MNLogit(y, X)
    try:
        fit_mn = mn.fit(method="newton", maxiter=200, disp=False)
    except Exception as exc:
        if verbose:
            print(f"MNLogit newton failed ({exc}); retrying bfgs")
        fit_mn = mn.fit(method="bfgs", maxiter=500, disp=False)
    if verbose:
        print(fit_mn.summary())

    cats = np.sort(y.unique())
    probs = fit_mn.predict(X)
    if isinstance(probs, pd.DataFrame):
        prob_df = probs.copy()
        prob_df.index = gps.index
        if set(prob_df.columns) != set(cats) and prob_df.shape[1] == len(cats):
            prob_df.columns = cats
    else:
        probs = np.asarray(probs, dtype=float)
        if probs.ndim == 1:
            raise RuntimeError("MNLogit.predict returned a vector; expected (n, J) probabilities")
        prob_df = pd.DataFrame(probs, index=gps.index, columns=cats)
    if 0 not in prob_df.columns:
        raise RuntimeError(f"MNLogit probabilities missing control column 0; columns={list(prob_df.columns)}")
    p0 = prob_df[0].astype(float).clip(lower=1e-6, upper=1).to_numpy()
    col_pos = {c: j for j, c in enumerate(prob_df.columns)}
    pk = prob_df.to_numpy()[np.arange(len(gps)), [col_pos[int(t)] for t in y]]
    pk = np.clip(pk.astype(float), 1e-6, 1.0)
    t = y.to_numpy()
    gps["ipw_weight"] = np.where(t == 0, 1.0, p0 / pk).astype(float)
    gps["gps_w"] = gps["ipw_weight"]
    if verbose:
        for k in cats:
            w = gps.loc[gps[bin_col] == k, "ipw_weight"]
            ess = float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else np.nan
            print(f"Bin {k}: n={len(w):,}  weight min={w.min():.3f}  "
                  f"mean={w.mean():.3f}  max={w.max():.3f}  ESS={ess:.1f}")

    rows = []
    for k in (1, 2):
        msk_t = gps[bin_col] == k
        msk_c = gps[bin_col] == 0
        for col in gps_x:
            x_t = gps.loc[msk_t, col].astype(float)
            x_c = gps.loc[msk_c, col].astype(float)
            smd_b = _weighted_smd(x_t, x_c)
            smd_a = _weighted_smd(
                x_t, x_c,
                gps.loc[msk_t, "ipw_weight"], gps.loc[msk_c, "ipw_weight"],
            )
            rows.append({
                "bin": k,
                "covariate": col,
                "label": GPS_X_LABELS.get(col, col),
                "smd_before": smd_b,
                "smd_after": smd_a,
                "abs_after": abs(smd_a) if np.isfinite(smd_a) else np.nan,
            })
    bal_plot = pd.DataFrame(rows)
    max_abs = float(bal_plot["abs_after"].max()) if len(bal_plot) else np.nan
    if verbose:
        print("Weighted SMD (treated bin − control) before vs after IPW:")
        print(bal_plot.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    if np.isfinite(max_abs) and max_abs > 0.1:
        print(
            f"WARNING: Matching failed! Covariates are not sufficiently balanced. "
            f"Max absolute correlation/SMD is {max_abs:.3f}."
        )
    elif verbose:
        print(f"All covariates have |weighted SMD| ≤ 0.1 after IPW "
              f"(max |SMD| = {max_abs:.3f}).")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
        for ax, k in zip(axes, (1, 2)):
            sub = bal_plot[bal_plot["bin"] == k].reset_index(drop=True)
            yv = np.arange(len(sub))
            ax.scatter(sub["smd_before"], yv, marker="s", s=55, color="#a0aec0",
                       label="Before IPW", zorder=3)
            ax.scatter(sub["smd_after"], yv, marker="o", s=55, color="#2b6cb0",
                       label="After IPW", zorder=3)
            for i, r in sub.iterrows():
                ax.plot([r["smd_before"], r["smd_after"]], [i, i],
                        color="#cbd5e0", lw=1.2, zorder=1)
            ax.axvline(0, color="black", lw=0.8)
            ax.axvline(0.1, color="#c53030", ls="--", lw=0.8, label="|SMD|=0.1")
            ax.axvline(-0.1, color="#c53030", ls="--", lw=0.8)
            ax.set_yticks(yv)
            ax.set_yticklabels(sub["label"])
            ax.set_xlabel("Standardised mean difference (bin − control)")
            ax.set_title(f"Bin {k} vs control")
            ax.set_ylim(len(sub) - 0.5, -0.5)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(
            "Covariate balance before and after multinomial ATT IPW", y=1.02
        )
        plt.tight_layout()
        plt.show()
        if verbose:
            print(
                f"**Fig. 2b.** Standardised mean differences of GPS covariates between "
                f"each `tp_bin` and the zero-intensity control, before vs after ATT IPW. "
                f"Grey squares = unweighted; blue circles = IPW. *Notes:* Unit = home "
                f"statistical area; n = {len(gps):,} covariate-complete SAs; X = "
                f"{', '.join(gps_x)} (z-scored in the multinomial logit). Dashed lines at "
                f"|SMD| = 0.1. High-dose cut = {p75:.3f} detected TPs per 1,000 "
                f"(75th percentile of non-zero `{intensity_col}`)."
            )

    flag_col = _cfg("TREAT_FLAG")
    long_gps = build_long(gps, treat_col=flag_col if flag_col in gps.columns else None)
    _attach = gps[["yishuv_stat_2022", bin_col, "ipw_weight", "gps_w"]].drop_duplicates(
        "yishuv_stat_2022"
    )
    long_gps = long_gps.drop(
        columns=[c for c in (bin_col, "ipw_weight", "gps_w") if c in long_gps.columns]
    )
    long_gps = long_gps.merge(_attach, on="yishuv_stat_2022", how="inner")
    did_wins = _did_window_names(long_gps["win"])
    if verbose:
        print(f"long_gps: {len(long_gps):,} rows, "
              f"{long_gps['yishuv_stat_2022'].nunique():,} SAs, windows={did_wins}")

    did = fit_did_table(
        long_gps, outcomes, wins=did_wins,
        bin_col=bin_col, weights_col="ipw_weight",
    )
    _order = [lab for _, lab in outcomes if lab in set(did["label"])]
    _show = [c for c in ["label", "intensity", "bin", "coef", "se", "p", "sig",
                         "pre_mean_treated", "pct_of_pre", "wald_eq_p",
                         "n_obs", "n_sa", "n_clusters"] if c in did.columns]
    if verbose:
        print("Treatment: 1{tp_bin=1} × post  and  1{tp_bin=2} × post  |  weights: ipw_weight")
        for _win in did_wins:
            print(f"\n=== IPW-weighted WLS DiD, window: {_win} ===")
            print(did[did["win"] == _win][_show]
                  .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

        def _star_cell(r):
            mark = r["sig"] if isinstance(r.get("sig"), str) else stars(r["p"])
            return f"{r['coef']:.3f}{mark}" if mark else "--"

        _star = did.copy()
        _star["_cell"] = [_star_cell(r) for _, r in _star.iterrows()]
        print("\n=== DiD (significant coef × post; else --) ===")
        for k, labk in [(1, "Low/medium dose (bin 1)"), (2, "High dose (bin 2)")]:
            sub = _star[_star["bin"] == k] if "bin" in _star.columns else _star
            if sub.empty:
                continue
            wide = (
                sub.pivot_table(index="label", columns="win", values="_cell", aggfunc="first")
                .reindex(index=_order)
                .reindex(columns=did_wins)
                .fillna("--")
            )
            print(f"\n  {labk} × post")
            print(wide.to_string())
        if "wald_eq_p" in did.columns:
            wald = (
                did.drop_duplicates(["label", "win"])
                .pivot(index="label", columns="win", values="wald_eq_p")
                .reindex(index=_order)
                .reindex(columns=did_wins)
            )
            print("\n=== Wald p-value (H0: δ_bin1 = δ_bin2) ===")
            print(wald.round(3).to_string())
        print(f"\nSpecs: {len(did)}  (expected {len(outcomes) * len(did_wins) * 2} = "
              f"{len(outcomes)} outcomes × {len(did_wins)} windows × 2 bins)")
        print(
            f"\n**Table 1b.** Two-way fixed-effects WLS difference-in-differences: "
            f"`1{{tp_bin=1}} × post` and `1{{tp_bin=2}} × post` on weekday local-share "
            f"outcomes, with multinomial ATT IPW weights. *Notes:* Unit = home statistical "
            f"area × month; SA and month fixed effects; standard errors clustered on "
            f"locality; n = {long_gps['yishuv_stat_2022'].nunique():,} IPW-weighted SAs "
            f"(covariate-complete `panel_m`, not 1:1 matched pairs); "
            f"pre = {_cfg('PRE_LABEL')}, post = {_cfg('POST_LABEL')}. "
            f"GPS is MNLogit of `tp_bin` on z-scored X = {', '.join(gps_x)}; "
            f"high-dose cut = {p75:.3f} per 1,000 (p75 of non-zero `{intensity_col}`). "
            f"Wald tests H0: δ1 = δ2. Shares are fractions, so 0.01 = 1 pp. "
            f"Windows: {', '.join(did_wins)}."
        )

    if plot:
        win_plot = [w for w in did_wins if w in set(did["win"])]
        n_out = len(outcomes)
        n_col = 4
        n_row = int(np.ceil(n_out / n_col))
        fig, axes = plt.subplots(n_row, n_col, figsize=(12.5, 3.1 * n_row), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        _term_col = {1: "#2b6cb0", 2: "#c05621"}
        _term_lab = {1: "Bin 1 × post", 2: "Bin 2 × post"}
        for ax, (col, title) in zip(axes, outcomes):
            yv = np.arange(len(win_plot))
            ax.axvline(0, ls="--", color="#dd6b20", lw=1.1)
            for k in (1, 2):
                sub = (
                    did[(did["outcome"] == col) & (did["bin"] == k)]
                    .set_index("win")
                    .reindex(win_plot)
                )
                off = (k - 1.5) * 0.22
                colr = _term_col[k]
                ax.errorbar(
                    sub["coef"], yv + off,
                    xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                    fmt="none", ecolor=colr, elinewidth=1.3, capsize=3,
                )
                ax.scatter(sub["coef"], yv + off, s=40, c=colr, zorder=3,
                           label=_term_lab[k])
            ax.set_yticks(yv)
            ax.set_yticklabels(win_plot)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("DiD coef (bin × post)")
            ax.tick_params(labelsize=8)
            ax.invert_yaxis()
        for ax in axes[n_out:]:
            ax.set_visible(False)
        axes[0].set_ylabel("Time-of-day window")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_term_col[k], markersize=7, label=_term_lab[k])
            for k in (1, 2)
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
        plt.suptitle("IPW-weighted bin × post by window", y=1.02)
        plt.tight_layout()
        plt.show()
        if verbose:
            print(
                f"**Fig. 2c.** `tp_bin=1` × post (blue) and `tp_bin=2` × post (orange) "
                f"coefficients with 95% cluster-robust CIs, multinomial ATT-IPW WLS. "
                f"*Notes:* Same TWFE as Table 1b; windows {', '.join(win_plot)}; "
                f"sample is covariate-complete SAs with IPW weights, not the binary "
                f"1:1 matched pairs from Fig. 2. Wald tests whether the two bin "
                f"coefficients are equal."
            )

    out = MnipwDidResult(
        intensity_col=intensity_col, bin_col=bin_col, p75=p75,
        gps=gps, fit_mn=fit_mn, long_gps=long_gps, did=did,
        bal_plot=bal_plot, gps_x=gps_x, max_abs_smd=max_abs,
    )
    if set_globals:
        _write(
            TP_BIN_P75=p75,
            gps=gps,
            fit_mn=fit_mn,
            long_gps=long_gps,
            did_gps=did,
            bal_plot=bal_plot,
            DID_WINS=did_wins,
            BIN_COL=bin_col,
        )
    return out

class MnipwDidResult(SimpleNamespace):
    def __repr__(self):
        n_sa = 0
        if getattr(self, "gps", None) is not None and "yishuv_stat_2022" in self.gps.columns:
            n_sa = int(self.gps["yishuv_stat_2022"].nunique())
        n_did = 0 if getattr(self, "did", None) is None else len(self.did)
        return f"MnipwDidResult(bin_col={self.bin_col!r}, n_sa={n_sa:,}, did_rows={n_did})"

def _weighted_smd(x_t, x_c, w_t=None, w_c=None):
    """(weighted mean_t − weighted mean_c) / unweighted pooled SD."""
    x_t = np.asarray(x_t, dtype=float)
    x_c = np.asarray(x_c, dtype=float)
    w_t = np.ones(len(x_t)) if w_t is None else np.asarray(w_t, dtype=float)
    w_c = np.ones(len(x_c)) if w_c is None else np.asarray(w_c, dtype=float)
    ok_t = np.isfinite(x_t) & np.isfinite(w_t) & (w_t > 0)
    ok_c = np.isfinite(x_c) & np.isfinite(w_c) & (w_c > 0)
    x_t, w_t = x_t[ok_t], w_t[ok_t]
    x_c, w_c = x_c[ok_c], w_c[ok_c]
    if len(x_t) < 2 or len(x_c) < 2:
        return np.nan
    mt = np.average(x_t, weights=w_t)
    mc = np.average(x_c, weights=w_c)
    nt, nc = len(x_t), len(x_c)
    sp2 = ((nt - 1) * np.var(x_t, ddof=1) + (nc - 1) * np.var(x_c, ddof=1)) / (nt + nc - 2)
    if not np.isfinite(sp2) or sp2 <= 0:
        return 0.0
    return (mt - mc) / np.sqrt(sp2)

def event_study(data, y, cluster_col="semel_yishuv", ref=None, treat_col=None,
                weights_col=None):
    """Interact continuous intensity with every month except the reference month."""
    ref = _cfg("PRE_TS") if ref is None else ref
    treat_col = _cfg("INTENSITY_COL") if treat_col is None else treat_col
    need = [y, treat_col, "month", "yishuv_stat_2022", cluster_col]
    if weights_col:
        need.append(weights_col)
    d = data.dropna(subset=need).copy()
    terms = []
    for mm in sorted(d["month"].unique()):
        if mm == ref:
            continue
        name = "tr_" + pd.Timestamp(mm).strftime("%Y_%m")
        d[name] = d[treat_col].astype(float) * (d["month"] == mm).astype(int)
        terms.append((name, pd.Timestamp(mm)))
    if not terms:
        return pd.DataFrame()
    res = _fit_absorb(
        d, y, [t for t, _ in terms],
        ["yishuv_stat_2022", "month_str"], cluster_col,
        weights_col=weights_col,
    )
    if res is None:
        return pd.DataFrame()
    out = [_row(res, t, {"outcome": y, "month": mm}) for t, mm in terms]
    out.append({"outcome": y, "month": ref, "coef": 0.0, "se": 0.0, "t": np.nan,
                "p": np.nan, "ci_low": 0.0, "ci_high": 0.0, "n_obs": int(res.nobs)})
    return pd.DataFrame(out).sort_values("month")

def _event_tick(v):
    ts = pd.Timestamp(v)
    return str(_cfg("PLACEBO_YEAR")) if ts == _cfg("PLACEBO_TS") else ts.strftime("%b %Y")

