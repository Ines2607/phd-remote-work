import numpy as np
import pandas as pd
import pytest

from remote_work_utils import place2vec as p2v
from remote_work_utils import third_places as tp
from remote_work_utils.settings import load_settings

TP = load_settings("configs/third_places.json")
P2V = load_settings("configs/place2vec.json")
CALENDAR = load_settings("configs/template.json")["calendar"]


def test_tail_flags_require_all_outstanding_and_no_evening():
    params = {
        "n_std": 1.0,
        "outstanding_scores": ["score_weekday"],
        "not_outstanding_scores": ["score_evening"],
        "low_scores": ["score_home"],
    }
    df = pd.DataFrame(
        {
            "score_weekday": [0, 0, 0, 0, 10, 10],
            "score_evening": [0, 0, 0, 0, 0, 10],
            "score_home": [5, 5, 5, 5, -10, -10],
        }
    )
    flags = tp.find_candidates(df, params)
    assert flags["is_candidate_all"].tolist() == [False, False, False, False, True, False]


def test_work_schedule_classes():
    users = pd.DataFrame(
        {
            "total_work_days": [10, 10, 10],
            "office_day_counts": [2, 9, np.nan],
            "remote_work_days_min_work": [6, 0, 1],
            "flag_remote_work_smoothed_max_thresh": [0.6, 0.0, 0.1],
        }
    )
    assert tp.work_schedule(users, TP["work_schedule"]).tolist() == [
        "remote_worker",
        "onsite_worker",
        "neither_onsite_not_remote",
    ]


def test_poi_visit_stats_long_late_morning_share():
    start = pd.to_datetime(["2023-03-05 10:30", "2023-03-05 20:00", "2023-03-10 11:00", "2023-03-06 10:15"])
    visits = pd.DataFrame(
        {
            "poi_id": ["a", "a", "a", "a"],
            "identifier": ["u1", "u2", "u1", "u3"],
            "start_date_time": start,
            "date": start.normalize(),
            "duration_hours": [2.0, 2.0, 2.0, 0.5],
            "dist_home": [500, 2000, 500, 300],
            "dist_work": [3000, 3000, 3000, 3000],
            "after_home": [True, False, False, True],
            "after_work": [False, False, False, False],
            "prev_flag_work": [False, False, False, False],
        }
    )
    stats = tp.poi_visit_stats(visits, TP["visit_stats"]).iloc[0]
    # Sunday 10:30 long visit counts; Friday and short visits do not.
    assert stats["n_long_morning_weekday_visits"] == 1
    assert stats["share_friday"] == 0.25
    assert stats["rw_share"] == 0.25
    assert stats["share_dwells_home_lt_1km"] == 0.75


def test_flag_category():
    flags = tp.flag_category(pd.Series(["Cafe", "office", "gym"]), TP["poi_flags"])
    assert flags.tolist() == [True, False, pd.NA]


def test_elkan_noto_recovers_separable_positives():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": np.r_[rng.normal(2, 0.5, 100), rng.normal(-2, 0.5, 100)]})
    labelled = np.r_[(rng.random(100) < 0.4).astype(int), np.zeros(100, dtype=int)]
    pu = TP["pu_model"]
    model = tp.make_model(1.0, pu["l1_ratio"], pu["max_iter"], pu["random_state"]).fit(X, labelled)
    p = model.predict_proba(X)
    assert p[:100].mean() > 0.8 and p[100:].mean() < 0.2


def _visits(rows):
    df = pd.DataFrame(rows, columns=["identifier", "start", "hours", "flag_home", "flag_work", "poi_id", "geohash"])
    df["start_date_time"] = pd.to_datetime(df["start"])
    df["end_date_time"] = df["start_date_time"] + pd.to_timedelta(df["hours"], unit="h")
    df["date"] = df["start_date_time"].dt.normalize()
    df["duration_hours"] = df["hours"]
    df["hour"] = df["start_date_time"].dt.hour
    return df.assign(
        dwell_geohash=df["geohash"],
        pois_category="cafe",
        census_id_filled="1001",
        home_census="1000",
        work_census="1002",
    )


def test_sentences_split_sessions_and_collapse_repeats():
    visits = _visits(
        [
            ("u1", "2023-03-05 07:00", 1, True, False, None, "sv8wrqf"),
            ("u1", "2023-03-05 08:10", 1, True, False, None, "sv8wrqf"),
            ("u1", "2023-03-05 10:00", 3, False, False, "p1", "sv8wrqg"),
            ("u1", "2023-03-05 18:00", 2, True, False, None, "sv8wrqf"),
            ("u1", "2023-03-10 10:00", 3, False, False, "p1", "sv8wrqg"),
        ]
    )
    config = {**P2V, "corpus": {**P2V["corpus"], "min_poi_count": 1}}
    sentences = p2v.build_sentences(visits, config, CALENDAR, scheme=3)
    # Friday is not a workday; the 18:00 visit starts a new session (gap > 3h).
    assert sentences == [["HOME_1000", "POI_p1_WEEKDAY_LATE_MORNING"], ["HOME_1000"]]


def test_scheme0_tokens():
    visits = _visits([("u1", "2023-03-05 10:00", 1, False, True, None, "g"), ("u1", "2023-03-05 12:00", 1, False, False, "p", "g")])
    assert p2v.tokenize_visits(visits, P2V, scheme=0).tolist() == ["WORK", "CAT_cafe"]


def test_procrustes_alignment_recovers_rotation():
    pytest.importorskip("gensim")
    pytest.importorskip("scipy")
    from gensim.models import KeyedVectors

    rng = np.random.default_rng(1)
    tokens = [f"HOME_{i}" for i in range(20)]
    vectors = rng.normal(size=(20, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))

    class Model:
        def __init__(self, v):
            self.wv = KeyedVectors(5)
            self.wv.add_vectors(tokens, v)

    reference, rotated = Model(vectors), Model(vectors @ rotation)
    aligned, _ = p2v.align_procrustes(reference, rotated, tokens, P2V["alignment"]["min_anchor"])
    assert np.allclose(aligned.wv[tokens[0]], vectors[0], atol=1e-5)
