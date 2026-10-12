"""Word2Vec training and Procrustes alignment of embeddings across periods."""

from __future__ import annotations

import copy
from typing import Any, Iterable

import numpy as np


def train_word2vec(sentences: Iterable[list[str]], params: dict[str, Any]):
    """Skip-gram Word2Vec with the ``word2vec`` settings (gensim)."""
    from gensim.models import Word2Vec

    return Word2Vec(sentences=sentences, **params)


def shared_home_work_anchors(models: list) -> list[str]:
    """HOME_* / WORK_* tokens present in every model (default alignment anchors)."""
    shared = set.intersection(*(set(model.wv.key_to_index) for model in models))
    return sorted(token for token in shared if token.startswith(("HOME_", "WORK_")))


def align_procrustes(model_source, model_target, anchor_tokens: list[str], min_anchor: int):
    """Rotate ``model_target`` onto ``model_source`` using shared anchor tokens.

    Returns a rotated copy of the target model and the rotation matrix.
    """
    from scipy.linalg import orthogonal_procrustes

    source, target = model_source.wv, model_target.wv
    anchors = [t for t in anchor_tokens if t in source.key_to_index and t in target.key_to_index]
    if len(anchors) < min_anchor:
        raise ValueError(f"Only {len(anchors)} shared anchors (need >= {min_anchor})")
    a = np.array([source[t] for t in anchors])
    b = np.array([target[t] for t in anchors])
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    rotation, _ = orthogonal_procrustes(b, a)

    aligned = copy.deepcopy(model_target)
    aligned.wv.vectors = aligned.wv.vectors @ rotation
    if hasattr(aligned, "syn1neg"):
        aligned.syn1neg = aligned.syn1neg @ rotation
    return aligned, rotation
