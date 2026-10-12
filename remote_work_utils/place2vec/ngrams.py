"""Frequent visit sequences (n-grams) in a place2vec corpus.

Tokens can be generalised before counting, e.g. ``POI_<id>`` ->
``POI_<neighbourhood type>_<category>_<status>``, using lookups passed in by
the caller (no database access).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TokenLookups:
    """Optional metadata used to generalise tokens.

    ``area_type``: area id -> neighbourhood type (for HOME_/WORK_ tokens);
    ``poi_area_type``, ``poi_category``, ``poi_context``, ``poi_status``: poi id -> label;
    ``geohash_context``: geohash -> context type (for GH_ tokens).
    """

    area_type: dict[str, str] = field(default_factory=dict)
    poi_area_type: dict[str, str] = field(default_factory=dict)
    poi_category: dict[str, str] = field(default_factory=dict)
    poi_context: dict[str, str] = field(default_factory=dict)
    poi_status: dict[str, str] | None = None
    geohash_context: dict[str, str] = field(default_factory=dict)


def parse_token(token: str) -> dict:
    """Split a token into kind (HOME/WORK/POI/GH/...), area, POI id, geohash and period."""
    parsed = {"raw": token, "kind": None, "area": None, "poi_id": None, "geohash": None, "period": None}
    if token.startswith(("HOME", "WORK")):
        parts = token.split("_")
        parsed["kind"] = parts[0]
        parsed["area"] = parts[1] if len(parts) >= 2 else None
    elif token.startswith("POI_"):
        tail = token[len("POI_"):]
        parsed["kind"] = "POI"
        parsed["poi_id"], parsed["period"] = tail.split("_WEEKDAY_", 1) if "_WEEKDAY_" in tail else (tail, None)
    elif token.startswith("GH_"):
        tail = token[3:]
        parsed["kind"] = "GH"
        parsed["geohash"], parsed["period"] = tail.split("_", 1) if "_" in tail else (tail, None)
    else:
        parsed["kind"] = token.split("_", 1)[0] or "OTHER"
    return parsed


def _label(value) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return text if text and text.lower() not in {"nan", "none", "na"} else None


def generalise_token(parsed: dict, lookups: TokenLookups, by_area_type: bool, by_category: bool, by_context: bool) -> str:
    """``BASE[_area type][_category][_context][_status][_period]``."""
    kind = parsed["kind"]
    if kind not in {"HOME", "WORK", "POI", "GH"}:
        return parsed["raw"]
    parts = [kind]
    poi_id = parsed["poi_id"]
    if by_area_type:
        area_type = lookups.area_type.get(parsed["area"]) if kind in {"HOME", "WORK"} else lookups.poi_area_type.get(poi_id)
        if area_type:
            parts.append(str(area_type))
    if by_category and kind == "POI":
        category = _label(lookups.poi_category.get(poi_id))
        if category:
            parts.append(category)
    if by_context:
        context = lookups.poi_context.get(poi_id) if kind == "POI" else lookups.geohash_context.get(parsed["geohash"])
        if context:
            parts.append(str(context))
    if kind == "POI" and lookups.poi_status is not None:
        parts.append(lookups.poi_status.get(poi_id, "unlabelled"))
    if parsed["period"]:
        parts.append(f"WEEKDAY_{parsed['period']}" if kind == "POI" else parsed["period"])
    return "_".join(parts)


def top_ngrams(
    sentences: Iterable[list[str]],
    n: int,
    top_k: int,
    lookups: TokenLookups | None = None,
    by_area_type: bool = False,
    by_category: bool = False,
    by_context: bool = False,
    exclude_prefixes: tuple[str, ...] = (),
    split_by_home_type: bool = False,
):
    """Most frequent n-grams.

    Returns ``[(ngram, percent of all n-grams)]``, or, with
    ``split_by_home_type``, ``{home type: [(ngram, count per sentence)]}``
    using the area type of the first HOME token in each n-gram.
    """
    lookups = lookups or TokenLookups()
    generalise = by_area_type or by_category or by_context or lookups.poi_status is not None
    counter: Counter = Counter()
    by_home: dict[str, Counter] = defaultdict(Counter)
    sentences_by_home: dict[str, set] = defaultdict(set)

    for sentence in sentences:
        sentence = list(sentence)
        if split_by_home_type:
            for token in sentence:
                parsed = parse_token(token)
                if parsed["kind"] == "HOME" and lookups.area_type.get(parsed["area"]):
                    sentences_by_home[lookups.area_type[parsed["area"]]].add(tuple(sentence))
                    break
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            if exclude_prefixes and all(token.startswith(exclude_prefixes) for token in ngram):
                continue
            parsed = [parse_token(token) for token in ngram]
            if generalise:
                ngram = tuple(generalise_token(p, lookups, by_area_type, by_category, by_context) for p in parsed)
            if split_by_home_type:
                home_type = next(
                    (lookups.area_type.get(p["area"]) for p in parsed if p["kind"] == "HOME"), None
                )
                if home_type:
                    by_home[home_type][ngram] += 1
            else:
                counter[ngram] += 1

    if split_by_home_type:
        return {
            home: [(ngram, count / max(len(sentences_by_home[home]), 1)) for ngram, count in counts.most_common(top_k)]
            for home, counts in by_home.items()
        }
    total = sum(counter.values())
    return [(ngram, 100.0 * count / total) for ngram, count in counter.most_common(top_k)] if total else []
