"""Positive-unlabelled (PU) classification of remote-work third places.

Elkan & Noto (2008) with an elastic-net logistic regression, as in
``800-research-users-locations/3000-pu-modelling``. Known positives are places
with remote-work evidence in reviews; all other places are unlabelled.
Validation hides a share of the positives and checks how highly they rank.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ElkanNotoPU:
    """P(y=1|x) = P(s=1|x) / c, with c = mean P(s=1|x) over labelled positives."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.c_ = None

    def fit(self, X, s):
        s = np.asarray(s)
        self.estimator.fit(X, s)
        self.c_ = float(np.mean(self.estimator.predict_proba(X)[:, 1][s == 1]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.c_ is None:
            raise ValueError("The model is not fitted yet")
        return np.clip(self.estimator.predict_proba(X)[:, 1] / self.c_, 0.0, 1.0)

    def predict(self, X, threshold: float) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def make_model(C: float, l1_ratio: float, max_iter: int, random_state: int) -> ElkanNotoPU:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn import __version__ as sklearn_version
    from sklearn.preprocessing import StandardScaler

    # scikit-learn >= 1.8 infers the elastic-net penalty from l1_ratio.
    major, minor = (int(part) for part in sklearn_version.split(".")[:2])
    penalty = {} if (major, minor) >= (1, 8) else {"penalty": "elasticnet"}
    logistic = LogisticRegression(
        l1_ratio=l1_ratio, C=C, solver="saga", max_iter=max_iter, random_state=random_state, **penalty
    )
    return ElkanNotoPU(Pipeline([("scaler", StandardScaler()), ("logreg", logistic)]))


def impute(features: pd.DataFrame, strategies: dict[str, str]) -> pd.DataFrame:
    """Fill missing values per column with ``mean`` or ``zero``."""
    out = features.copy()
    for column, strategy in strategies.items():
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if strategy == "mean":
            out[column] = values.fillna(values.mean())
        elif strategy == "zero":
            out[column] = values.fillna(0)
        else:
            raise ValueError(f"Unknown imputation strategy '{strategy}' for {column}")
    return out


def coefficients(model: ElkanNotoPU, feature_names: list[str]) -> pd.DataFrame:
    """Standardised logistic coefficients, largest absolute first."""
    coef = model.estimator.named_steps["logreg"].coef_.ravel()
    return (
        pd.DataFrame({"feature": feature_names, "coef_scaled": coef, "abs_coef": np.abs(coef)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )


def _hide_positives(y: np.ndarray, holdout_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positives = np.flatnonzero(y == 1)
    hidden = rng.choice(positives, size=max(1, int(round(holdout_frac * len(positives)))), replace=False)
    observed = y.copy()
    observed[hidden] = 0
    return hidden, observed


def positive_holdout_validation(X: pd.DataFrame, y, params: dict[str, Any], C: float, l1_ratio: float) -> pd.DataFrame:
    """Repeatedly hide positives, retrain and record where they rank (recall@K, median percentile)."""
    validation = params["validation"]
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    rng = np.random.default_rng(params["random_state"])
    rows = []
    for repeat in range(validation["n_repeats"]):
        hidden, observed = _hide_positives(y, validation["holdout_frac"], rng)
        model = make_model(C, l1_ratio, params["max_iter"], params["random_state"] + repeat).fit(X, observed)
        scores = model.predict_proba(X)
        ranking = np.argsort(-scores)
        position = np.empty_like(ranking)
        position[ranking] = np.arange(1, len(scores) + 1)
        row = {
            "repeat": repeat,
            "c_hat": model.c_,
            "hidden_positive_median_rank": float(np.median(position[hidden])),
            "hidden_positive_median_percentile": float(np.median(1 - (position[hidden] - 1) / len(scores))),
            "hidden_positive_mean_score": float(scores[hidden].mean()),
            "unlabeled_mean_score": float(scores[y == 0].mean()),
        }
        for k in validation["top_k_values"]:
            row[f"recall_at_{k}"] = float(np.isin(hidden, ranking[:k]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def hidden_positive_enrichment(X: pd.DataFrame, y, params: dict[str, Any]) -> pd.DataFrame:
    """Share of hidden positives recovered in the top x% of places, over the C x l1 grid."""
    validation = params["validation"]
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    rng = np.random.default_rng(params["random_state"])
    shares = np.arange(0, 1.0 + validation["top_share_step"] / 2, validation["top_share_step"])
    rows = []
    for repeat in range(validation["n_repeats"]):
        hidden, observed = _hide_positives(y, validation["holdout_frac"], rng)
        hidden_mask = np.zeros(len(y), dtype=bool)
        hidden_mask[hidden] = True
        baseline = len(hidden) / len(y)
        for C in validation["C_grid"]:
            for l1_ratio in validation["l1_ratio_grid"]:
                model = make_model(C, l1_ratio, params["max_iter"], params["random_state"] + repeat).fit(X, observed)
                ranking = np.argsort(-model.predict_proba(X))
                for share in shares:
                    k = max(1, int(round(share * len(y))))
                    found = int(hidden_mask[ranking[:k]].sum())
                    rows.append(
                        {
                            "repeat": repeat,
                            "C": C,
                            "l1_ratio": l1_ratio,
                            "top_share": share,
                            "top_k": k,
                            "share_of_hidden_positives_recovered": found / len(hidden),
                            "enrichment_over_baseline": (found / k) / baseline,
                        }
                    )
    return pd.DataFrame(rows)


def top_k_stability(X: pd.DataFrame, y, params: dict[str, Any], C: float, l1_ratio: float) -> dict[str, float]:
    """Mean pairwise Jaccard overlap of the top-K places across holdout repeats."""
    validation = params["validation"]
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    rng = np.random.default_rng(params["random_state"])
    top_sets = []
    for repeat in range(validation["n_repeats"]):
        _, observed = _hide_positives(y, validation["holdout_frac"], rng)
        model = make_model(C, l1_ratio, params["max_iter"], params["random_state"] + repeat).fit(X, observed)
        top_sets.append(set(np.argsort(-model.predict_proba(X))[: validation["stability_top_k"]]))
    jaccard = [
        len(a & b) / len(a | b) for i, a in enumerate(top_sets) for b in top_sets[i + 1 :] if a | b
    ]
    return {"mean_pairwise_jaccard": float(np.mean(jaccard)), "min_pairwise_jaccard": float(np.min(jaccard))}
