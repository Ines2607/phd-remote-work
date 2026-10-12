"""place2vec: Word2Vec embeddings of visit sequences (tokens = home, work, POIs, areas).

Requires the ``place2vec`` extra (gensim, scipy). See
``notebook_docs/methodology/PLACE2VEC.md``.
"""

from remote_work_utils.place2vec.corpus import build_sentences
from remote_work_utils.place2vec.ngrams import TokenLookups, top_ngrams
from remote_work_utils.place2vec.similarity import (
    avg_similarity_to_centroid,
    build_centroid,
    build_work_centroid,
    flag_daytime_work_places,
    poi_period_similarities,
)
from remote_work_utils.place2vec.tokenize import day_part, tokenize_visits
from remote_work_utils.place2vec.train import align_procrustes, shared_home_work_anchors, train_word2vec

__all__ = [
    "TokenLookups",
    "align_procrustes",
    "avg_similarity_to_centroid",
    "build_centroid",
    "build_sentences",
    "build_work_centroid",
    "day_part",
    "flag_daytime_work_places",
    "poi_period_similarities",
    "shared_home_work_anchors",
    "tokenize_visits",
    "top_ngrams",
    "train_word2vec",
]
