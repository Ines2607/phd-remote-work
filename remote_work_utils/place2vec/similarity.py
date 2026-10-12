"""Concept centroids (work, third place) and POI similarity profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_centroid(model, tokens: list[str]) -> tuple[np.ndarray, list[str]]:
    """Unit-norm mean vector of ``tokens`` present in the model."""
    present = [t for t in tokens if t in model.wv.key_to_index]
    if not present:
        raise ValueError("None of the tokens are in the model vocabulary")
    return _unit(np.mean([model.wv[t] for t in present], axis=0).astype(np.float64)), present


def build_work_centroid(model) -> tuple[np.ndarray, list[str]]:
    """``WORK`` token (scheme 0) or the mean of all ``WORK_*`` tokens."""
    if "WORK" in model.wv.key_to_index:
        return _unit(model.wv["WORK"].astype(np.float64)), ["WORK"]
    return build_centroid(model, [t for t in model.wv.key_to_index if t.startswith("WORK_")])


def avg_similarity_to_centroid(model, tokens: list[str], centroid: np.ndarray, min_similarity: float) -> tuple[float, list[str]]:
    """Mean cosine similarity of tokens to a centroid, ignoring those below ``min_similarity``."""
    present = [t for t in tokens if t in model.wv.key_to_index]
    if not present:
        return float("nan"), []
    sims = model.wv.cosine_similarities(centroid, [model.wv[t] for t in present])
    kept = [(t, s) for t, s in zip(present, sims) if s >= min_similarity]
    if not kept:
        return float("nan"), []
    return float(np.mean([s for _, s in kept])), [t for t, _ in kept]


def poi_period_similarities(model, centroids: dict[str, np.ndarray], periods: list[str]) -> pd.DataFrame:
    """Similarity of every ``POI_<id>_WEEKDAY_<period>`` token to each centroid.

    One row per POI; columns ``<period>_<centroid name>``.
    """
    rows: dict[str, dict[str, float]] = {}
    for token in model.wv.key_to_index:
        if not token.startswith("POI_") or "_WEEKDAY_" not in token:
            continue
        poi_id, period = token[len("POI_"):].split("_WEEKDAY_", 1)
        if period not in periods:
            continue
        row = rows.setdefault(poi_id, {"poi_id": poi_id})
        for name, centroid in centroids.items():
            row[f"{period}_{name}"] = float(model.wv.cosine_similarities(centroid, [model.wv[token]])[0])
    return pd.DataFrame(list(rows.values()))


def flag_daytime_work_places(similarities: pd.DataFrame, params: dict[str, Any], centroid_name: str = "WORK") -> pd.DataFrame:
    """POIs that look like work in the daytime but not in the evening.

    ``qualifies_remote_work`` = max daytime similarity > ``work_threshold`` and
    every evening/night similarity below the threshold or missing.
    """
    result = similarities.copy()
    day = [f"{p}_{centroid_name}" for p in params["daytime_periods"] if f"{p}_{centroid_name}" in result]
    evening = [f"{p}_{centroid_name}" for p in params["evening_periods"] if f"{p}_{centroid_name}" in result]
    result["max_daytime_work_sim"] = result[day].max(axis=1) if day else np.nan
    low_evening = pd.Series(True, index=result.index)
    for column in evening:
        low_evening &= result[column].isna() | (result[column] < params["work_threshold"])
    result["qualifies_remote_work"] = (result["max_daytime_work_sim"] > params["work_threshold"]) & low_evening
    return result


def closest_tokens(model, token: str, top_k: int) -> pd.DataFrame:
    return pd.DataFrame(model.wv.most_similar(token, topn=top_k), columns=["token", "similarity"])


def neighbour_category_agreement(model, category_tokens: dict[str, list[str]], top_k: int) -> pd.DataFrame:
    """Stability check: share of a token's top-K neighbours from the same category."""
    rows = []
    for category, tokens in category_tokens.items():
        for token in tokens:
            if token in model.wv.key_to_index:
                neighbours = {t for t, _ in model.wv.most_similar(token, topn=top_k)}
                rows.append({"category": category, "token": token, "matching_rate": len(neighbours & set(tokens)) / top_k})
    return pd.DataFrame(rows)


def rank_to_centroid(model, centroid: np.ndarray, tokens: list[str]) -> pd.DataFrame:
    """Rank of each token among all vocabulary items by similarity to ``centroid`` (1 = closest)."""
    ranking = model.wv.similar_by_vector(centroid, topn=len(model.wv))
    rank = {token: i + 1 for i, (token, _) in enumerate(ranking)}
    return pd.DataFrame({"token": tokens, "rank": [rank.get(t) for t in tokens]})
