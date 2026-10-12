import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
import re
import os
import io
import json
from collections import Counter, defaultdict
from datetime import timedelta
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine as cosine_dist
import holidays
from pathlib import Path

from geofunctions import utils

# Set up logging
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / Configuration
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG = json.loads((Path(__file__).with_name("place2vec_config.json")).read_text())

PROJECT = _CONFIG["project"]

# BQ table references
TABLE_DWELLS_RAW = _CONFIG["tables"]["dwells_raw"]
TABLE_WORK_LOC = _CONFIG["tables"]["work_locations"]
TABLE_GEOHASH_SA = _CONFIG["tables"]["geohash_stat_areas"]
TABLE_GOOGLE_POIS = _CONFIG["tables"]["google_pois"]
TABLE_OSM_POIS = _CONFIG["tables"]["osm_pois"]
TABLE_DWELLS_POIS = _CONFIG["tables"]["dwells_pois"]
TABLE_USER_PROFILES = _CONFIG["tables"]["user_profiles"]
TABLE_GEOHASH_CONTEXT = _CONFIG["tables"]["geohash_context"]
TABLE_GEOHASH_FEATURES = _CONFIG["tables"]["geohash_features"]
TABLE_GOOGLE_POIS_FEATURES = _CONFIG["tables"]["google_pois_features"]
TABLE_NEIGHBOURHOOD_TYPES = _CONFIG["tables"]["neighbourhood_types"]

# Period definitions (Israel COVID context)
PERIODS = _CONFIG["periods"]

# Israel-specific election days
ELECTION_DAYS = _CONFIG["election_days"]


def _build_israel_non_weekdays(years):
    """
    Build a set of dates that are NOT weekdays by the Israel calendar:
      - Friday (weekday=4) and Saturday (weekday=5)
      - Israeli public holidays
      - Erev chag (day before each holiday)
      - Election days

    Mirrors the flag_weekend logic from 002-work-location-monthly.py.

    Parameters
    ----------
    years : iterable of int   e.g. [2019, 2020, 2021, 2022, 2023]

    Returns
    -------
    set of datetime.date
    """
    non_weekdays = set()

    # holidays + erev chag
    for yr in years:
        il_holidays = holidays.country_holidays("IL", years=yr)
        for d in il_holidays.keys():
            non_weekdays.add(d)
            non_weekdays.add(d - timedelta(days=1))  # erev chag

    # election days
    for ed in ELECTION_DAYS:
        non_weekdays.add(pd.Timestamp(ed).date())

    # NOTE: Fridays & Saturdays are handled via weekday check at filter time,
    # not stored here, to keep the set small.
    return non_weekdays


def _is_weekday_israel(date_series, non_weekday_dates):
    """
    Return a boolean Series: True for Israel weekdays (Sun–Thu, excluding
    holidays / erev chag / elections).

    Parameters
    ----------
    date_series : pd.Series of datetime64
    non_weekday_dates : set of datetime.date
    """
    dt = pd.to_datetime(date_series)
    # Friday=4, Saturday=5 → weekend
    is_fri_sat = dt.dt.weekday.isin([4, 5])
    is_special = dt.dt.date.isin(non_weekday_dates)
    return ~(is_fri_sat | is_special)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 : Recreate dwells_pois_home_work
# ═══════════════════════════════════════════════════════════════════════════════

def create_dwells_pois_home_work(
    project=PROJECT,
    google_poi_radius_m=20,
    osm_poi_radius_m=20,
    table_suffix="_v2",
    month_list=None,
):
    """
    Recreate `phd_dwells.dwells_pois_home_work` with cascading POI join:
      1. Flag home (geohash_home == dwell_geohash) / work (geohash_work_office == dwell_geohash)
      2. Google 3rd-place POIs within *google_poi_radius_m*  (priority)
      3. OSM / Foursquare POIs within *osm_poi_radius_m*  (fallback)
      4. Drop dwells that match nothing
      5. Attach census_id + context_type via geohash
      6. Compute unified ``place_id``: POI id for POI matches, dwell_geohash otherwise

    Parameters
    ----------
    project : str
        GCP project id.
    google_poi_radius_m : float
        ST_DWITHIN radius (metres) for Google 3rd-place POI join.  Default 20.
    osm_poi_radius_m : float
        ST_DWITHIN radius (metres) for OSM / Foursquare fallback POI join.
        Default 20.
    table_suffix : str
        Suffix appended to the base table name so the original table is
        preserved.  Default "_v2"  →  ``dwells_pois_home_work_v2``.
    month_list : list[str] or None
        Optional.  If provided, only process these months (partition pruning).
        Format: ["2019-01-01", "2020-06-01", ...].  None = all months.

    Runs entirely in BigQuery (no local memory needed).
    """
    month_filter = ""
    if month_list:
        month_csv = ", ".join(f"DATE('{m}')" for m in month_list)
        month_filter = f"\n      AND DATE_TRUNC(d.date, MONTH) IN ({month_csv})"

    query = f"""
    CREATE OR REPLACE TABLE `{TABLE_DWELLS_POIS}{table_suffix}`
    PARTITION BY month
    CLUSTER BY identifier
    AS

    /* ── Step 1: base dwells + home / work geohashes ── */
    WITH dwells AS (
      SELECT
        d.identifier,
        d.timestamp,
        d.duration_hours,
        TIMESTAMP_MICROS(CAST(d.end_date_time / 1000 AS INT64)) AS end_date_time,
        d.start_date_time,
        d.date,
        d.geohash                            AS dwell_geohash,
        DATE_TRUNC(d.date, MONTH)            AS month,
        d.geometry,
        w.geohash_home,
        w.geohash_work_office,
        ST_CENTROID(gg.geometry)              AS work_centroid,
        ST_GEOGFROMTEXT(w.home_geometry)      AS home_geom
      FROM `{TABLE_DWELLS_RAW}` d
      JOIN `{TABLE_WORK_LOC}` w
        ON d.identifier = w.identifier
       AND DATE_TRUNC(d.date, MONTH) = w.month  /* month in join: one home/work per user per month */
      JOIN `{TABLE_GEOHASH_SA}` gg
        ON gg.geohash = w.geohash_work_office
      WHERE 1=1{month_filter}
    ),

    /* ── Step 2: Home/Work branch - geohash match, NO POI spatial joins ── */
    home_work AS (
      SELECT
        identifier, timestamp, duration_hours, end_date_time,
        start_date_time, date, month, dwell_geohash, geometry,
        geohash_home = dwell_geohash           AS flag_home,
        geohash_work_office = dwell_geohash    AS flag_work,
        geohash_home, geohash_work_office,
        CAST(NULL AS STRING)                  AS id,
        CAST(NULL AS STRING)                  AS pois_category,
        CAST(NULL AS STRING)                  AS keyword,
        CAST(NULL AS STRING)                  AS name,
        CAST(NULL AS STRING)                  AS poi_source,
        CAST(NULL AS FLOAT64)                 AS dist_to_poi,
        ST_DISTANCE(home_geom, geometry)      AS dist_to_home,
        ST_DISTANCE(work_centroid, geometry)  AS dist_to_work
      FROM dwells
      WHERE geohash_home = dwell_geohash OR geohash_work_office = dwell_geohash
    ),

    /* ── Step 3: Third-place candidates - run POI joins on this subset ── */
    third AS (
      SELECT *, FALSE AS flag_home, FALSE AS flag_work
      FROM dwells
      WHERE geohash_home != dwell_geohash AND geohash_work_office != dwell_geohash
    ),

    /* ── Step 4: Google 3rd-place POIs - INNER JOIN (never LEFT + ST_DWITHIN) ── */
    /* Geohash pre-filter + INNER JOIN so only matching pairs are produced */
    google_matched AS (
      SELECT
        d.*,
        gp.id                                AS google_poi_id,
        gp.primarytype                       AS google_primarytype,
        gp.displayname                       AS google_displayname,
        ST_DISTANCE(d.geometry, gp.geometry) AS dist_to_google_poi,
        ROW_NUMBER() OVER (
          PARTITION BY d.identifier, d.timestamp
          ORDER BY ST_DISTANCE(d.geometry, gp.geometry)
        ) AS rn_google
      FROM third d
      INNER JOIN `{TABLE_GOOGLE_POIS}` gp
        ON ST_DWITHIN(d.geometry, gp.geometry, {google_poi_radius_m})
    ),
    google_best AS (
      SELECT * FROM google_matched WHERE rn_google = 1
    ),

    /* ── Step 5: OSM fallback - dwells with no Google match, INNER JOIN POIs ── */
    osm_matched AS (
      SELECT
        d.*,
        osm.id                                AS osm_poi_id,
        osm.pois_category                     AS osm_pois_category,
        osm.keyword                           AS osm_keyword,
        osm.name                              AS osm_name,
        ST_DISTANCE(d.geometry, osm.geometry) AS dist_to_osm_poi,
        ST_DISTANCE(d.home_geom, d.geometry)  AS dist_to_home,
        ST_DISTANCE(d.work_centroid, d.geometry) AS dist_to_work,
        ROW_NUMBER() OVER (
          PARTITION BY d.identifier, d.timestamp
          ORDER BY ST_DISTANCE(d.geometry, osm.geometry)
        ) AS rn_osm
      FROM (
        SELECT d.* FROM third d
        LEFT JOIN (SELECT identifier, timestamp FROM google_best) gb
          ON d.identifier = gb.identifier AND d.timestamp = gb.timestamp
        WHERE gb.identifier IS NULL
      ) d
      INNER JOIN `{TABLE_OSM_POIS}` osm
        ON ST_DWITHIN(d.geometry, osm.geometry, {osm_poi_radius_m})
    ),
    osm_best AS (
      SELECT * FROM osm_matched WHERE rn_osm = 1
    ),

    /* ── Step 6: third_matched = Google matches + OSM matches (no Google) ── */
    third_matched AS (
      SELECT
        identifier, timestamp, duration_hours, end_date_time,
        start_date_time, date, month, dwell_geohash, geometry,
        flag_home, flag_work, geohash_home, geohash_work_office,
        google_poi_id AS id,
        google_primarytype AS pois_category,
        google_primarytype as keyword,
        google_displayname.text AS name,
        'google' AS poi_source,
        dist_to_google_poi AS dist_to_poi,
        ST_DISTANCE(home_geom, geometry) AS dist_to_home,
        ST_DISTANCE(work_centroid, geometry) AS dist_to_work
      FROM google_best
      UNION ALL
      SELECT
        identifier, timestamp, duration_hours, end_date_time,
        start_date_time, date, month, dwell_geohash, geometry,
        flag_home, flag_work, geohash_home, geohash_work_office,
        osm_poi_id AS id,
        osm_pois_category AS pois_category,
        osm_keyword AS keyword,
        osm_name AS name,
        'osm_frsq' AS poi_source,
        dist_to_osm_poi AS dist_to_poi,
        ST_DISTANCE(home_geom, geometry) AS dist_to_home,
        ST_DISTANCE(work_centroid, geometry) AS dist_to_work
      FROM osm_best
    ),

    /* ── Step 8: UNION home/work + third-place matched ── */
    merged AS (
      SELECT * FROM home_work
      UNION ALL
      SELECT * FROM third_matched
    )

    /* ── Step 9: attach census area, context, place_id, and compute lag / day-type columns ── */
    SELECT
      m.identifier,
      m.id,
      /* ── unified place_id: POI id when matched, otherwise dwell geohash ── */
      CASE
        WHEN m.id IS NOT NULL THEN CAST(m.id AS STRING)
        ELSE m.dwell_geohash
      END                                    AS place_id,
      m.dwell_geohash,
      m.geohash_home,
      m.geometry,
      m.timestamp,
      m.duration_hours,
      m.start_date_time,
      m.end_date_time,
      m.flag_home,
      m.flag_work,
      m.pois_category,
      m.keyword,
      m.poi_source,
      m.name,
      m.dist_to_home,
      m.dist_to_work,
      m.dist_to_poi,
      m.month,
      m.date,
      CASE
        WHEN EXTRACT(DAYOFWEEK FROM m.date) IN (1,2,3,4,5) THEN 'weekday'
        WHEN EXTRACT(DAYOFWEEK FROM m.date) = 6 THEN 'Friday'
        ELSE 'Saturday'
      END AS day_type,
      RANK() OVER (
        PARTITION BY m.identifier, m.date
        ORDER BY m.timestamp
      ) AS day_order,
      ST_DISTANCE(
        LAG(m.geometry) OVER (PARTITION BY m.identifier, m.date ORDER BY m.timestamp),
        m.geometry
      ) AS distance_from_prev,
      LAG(m.id) OVER (
        PARTITION BY m.identifier, m.date ORDER BY m.timestamp
      ) AS prev_poi_id,
      LAG(m.flag_home) OVER (
        PARTITION BY m.identifier, m.date ORDER BY m.timestamp
      ) AS prev_flag_home,
      LAG(m.flag_work) OVER (
        PARTITION BY m.identifier, m.date ORDER BY m.timestamp
      ) AS prev_flag_work,
      DATETIME_DIFF(
        LAG(m.end_date_time) OVER (PARTITION BY m.identifier, m.date ORDER BY m.timestamp),
        m.start_date_time, HOUR
      ) AS time_diff_from_prev,
      sa.yishuv_stat_2022                AS census_id,
      ctx.context_type
    FROM merged m
    LEFT JOIN `{TABLE_GEOHASH_SA}` sa
      ON m.dwell_geohash = sa.geohash
    LEFT JOIN `{TABLE_GEOHASH_CONTEXT}` ctx
      ON m.dwell_geohash = ctx.geohash
    """

    logger.info("Creating dwells_pois_home_work table in BQ ...")
    logger.info("Query:\n%s", query)
    utils.read_query_to_dataframe(query, project=project)
    logger.info("Done – table created.")


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy helpers (kept for backward-compat with monthly notebook)
# ═══════════════════════════════════════════════════════════════════════════════


def load_data_for_month(
    month,
    project=PROJECT,
    user_condition=None,
    load_static=True,
    load_users=True,
):
    """
    Loads dwells, and optionally static geohash features and user profiles.
    """
    print(f"Loading data for month: {month}")

    if user_condition is not None:
        user_condition = f"where {user_condition}"
    else:
        user_condition = ""
    USERS_PROFILES_TABLE = f"`{TABLE_USER_PROFILES}`"
    # 1. Dwells Query
    dwells_query = f"""
    SELECT *,
    case when EXTRACT(HOUR FROM start_date_time) between 6 and 10 then 'morning'
    when EXTRACT(HOUR FROM start_date_time) between 11 and 13
        then 'early_afternoon'
    when EXTRACT(HOUR FROM start_date_time) between 14 and 17
        then 'late_afternoon'
    when EXTRACT(HOUR FROM start_date_time) between 17 and 22
        then 'evening'
    else 'night' end as day_part,
    FROM `{TABLE_DWELLS_POIS}`
    WHERE month = '{month}' and identifier in (
    select identifier from {USERS_PROFILES_TABLE} 
     {user_condition})
    """  # remove where month = '{month}' to account for any office month
    df_dwells = utils.read_query_to_dataframe(dwells_query, project=project)

    # 2. Static Features Query
    df_static = None
    if load_static:
        static_query = f"""
        SELECT t.*, c.context_type
        FROM `{TABLE_GEOHASH_FEATURES}` t
        JOIN `{TABLE_GEOHASH_CONTEXT}` c
        USING (geohash)
        WHERE t.geohash IN (
            SELECT dwell_geohash
            FROM `{TABLE_DWELLS_POIS}`
            WHERE month = '{month}'
        )
        """
        df_static = utils.read_query_to_dataframe(static_query, project=project)

    # 3. User Profiles Query
    df_users = None
    if load_users:
        users_query = f"""
        SELECT * FROM `{TABLE_USER_PROFILES}`
        WHERE month='{month}'
        """
        df_users = utils.read_query_to_dataframe(users_query, project=project)

        # Clean User Flags
        flag_cols = [col for col in df_users.columns if col.startswith("FLAG_")]
        if flag_cols:
            df_users[flag_cols] = df_users[flag_cols].astype(bool)

    return df_dwells, df_static, df_users

# for geohashes only - not used
def calculate_column_thresholds(df, columns):
    """
    Calculates p25 and p75 for each column based on non-zero values.
    Returns: { 'cafe': {'low': 1, 'high': 5}, ... }
    """
    P25_PERCENTILE = 25
    P75_PERCENTILE = 75
    MIN_LOW_THRESHOLD = 1

    thresholds = {}
    print("Calculating dynamic thresholds per column...")

    for col in columns:
        if col not in df.columns:
            continue

        # Get only non-zero values
        non_zero_values = df[df[col] > 0][col]

        if len(non_zero_values) == 0:
            thresholds[col] = {"low": MIN_LOW_THRESHOLD, "high": MIN_LOW_THRESHOLD}
            continue

        # Calculate Percentiles
        p25 = np.percentile(non_zero_values, P25_PERCENTILE)
        p75 = np.percentile(non_zero_values, P75_PERCENTILE)

        low_thresh = max(MIN_LOW_THRESHOLD, int(p25))
        high_thresh = max(low_thresh, int(p75))

        thresholds[col] = {"low": low_thresh, "high": high_thresh}

    return thresholds


def process_static_attributes(df_static, poi_columns=None, str_type_columns=None):
    """
    Creates a dictionary of tags for every geohash using dynamic thresholds.
    """
    if df_static is None or df_static.empty:
        return {}

    FALLBACK_LOW = 1
    FALLBACK_HIGH = 3

    if poi_columns is None:
        exclude = ["geohash", "geometry"]
        poi_columns = [
            col
            for col in df_static.columns
            if col not in exclude and df_static[col].dtype in ["float64", "int64"]
        ]

    # Fill NA and set type for POI columns
    df_static[poi_columns] = df_static[poi_columns].fillna(0).astype(int)

    # 1. Pre-calculate thresholds for the dataset
    threshold_map = calculate_column_thresholds(df_static, poi_columns)

    print("Processing Static Attributes...")
    static_tags_map = {}

    for row in df_static.itertuples(index=False):
        gh_id = getattr(row, "geohash", None)
        if not gh_id:
            continue

        tags = []

        # Process numerical columns with thresholds
        for col in poi_columns:
            if hasattr(row, col):
                val = getattr(row, col)
                if pd.notna(val) and val > 0:
                    t = threshold_map.get(
                        col, {"low": FALLBACK_LOW, "high": FALLBACK_HIGH}
                    )
                    tag_base = f"ATTR_{col.upper()}"
                    if val >= t["high"]:
                        tags.append(f"{tag_base}_HIGH")
                    elif val <= t["low"]:
                        tags.append(f"{tag_base}_LOW")
                    else:
                        tags.append(f"{tag_base}_MED")

        # Process string columns directly (e.g., context_type)
        if str_type_columns:
            for col in str_type_columns:
                if hasattr(row, col):
                    val = getattr(row, col)
                    if pd.notna(val) and val != "":
                        tags.append(f"{col.upper()}_{str(val).upper()}")

        # Handle empty case
        if not tags:
            tags.append("ATTR_NO_POIS")

        static_tags_map[gh_id] = tags

    print(f"Finished processing attributes for {len(static_tags_map)} geohashes.")
    return static_tags_map


def build_corpus(
    df_dwells,
    static_tags_map,
    df_user=None,
    user_cols_general=None,
    user_cols_important=None,
    dwell_extra_columns=None,
):
    """
    Builds the trajectory corpus from dwells and static attributes.
    """
    # Day Part Thresholds
    MORNING_START = 6
    LATE_MORNING_START = 9
    AFTERNOON_START = 12
    EVENING_START = 17
    NIGHT_START = 23

    # Duration Thresholds (hours)
    DUR_SHORT_THRESH = 0.5
    DUR_MEDIUM_THRESH = 2.0
    DUR_LONG_THRESH = 4.0

    print("Building Trajectory Corpus...")

    # Ensure datetime and sorting
    df_dwells["start_date_time"] = pd.to_datetime(df_dwells["start_date_time"])
    df_dwells = df_dwells.sort_values(by=["identifier", "start_date_time"])

    if "date" not in df_dwells.columns:
        df_dwells["date"] = df_dwells["start_date_time"].dt.date

    # Helper function to convert row columns to tags
    def get_tags(row, cols, prefix="USER_"):
        tags = []
        if cols is None:
            return tags
        for col in cols:
            val = getattr(row, col, None)
            if pd.isna(val):
                continue
            # Handle flags (Booleans/Binary)
            is_bool_int = isinstance(val, (bool, np.bool_, int, np.integer))
            if is_bool_int and (val is True or val is False or val in [0, 1]):
                if val or val == 1:
                    tags.append(f"{prefix}{col.upper()}")
            # Handle categories (Strings/Ints)
            else:
                tags.append(f"{prefix}{col.upper()}_{str(val).upper()}")
        return tags

    # Prepare user metadata mapping for speed
    user_tags_map = {}
    if df_user is not None:
        df_u_lookup = df_user.set_index("identifier")
        for uid, row in df_u_lookup.iterrows():
            gen_tags = get_tags(row, user_cols_general, prefix="USER_")
            imp_tags = get_tags(row, user_cols_important, prefix="USER_")
            user_tags_map[uid] = (gen_tags, imp_tags)

    corpus = []
    # Group by user and date to form daily trajectories
    for (user_id, date_val), group in df_dwells.groupby(["identifier", "date"]):
        daily_trajectory = []

        # Get this user persistent tags
        u_gen_tags, u_imp_tags = user_tags_map.get(user_id, ([], []))

        # --- 1. General User Profile Tokens (Injected ONCE per day) ---
        daily_trajectory.extend(u_gen_tags)

        for row in group.itertuples(index=False):
            gh_id = getattr(row, "dwell_geohash", None)
            if not gh_id:
                continue

            # --- 2. Important User Profile Tokens (Injected for EVERY loc) ---
            daily_trajectory.extend(u_imp_tags)

            # --- 3. Anchor Token (Geohash + ID if id is in dwell_extra_columns) ---
            # Check if 'id' is in dwell_extra_columns and combine with geohash
            if dwell_extra_columns and "id" in dwell_extra_columns:
                id_val = getattr(row, "id", None)
                if pd.notna(id_val):
                    # Combined token: GH_{geohash}_ID_{id}
                    daily_trajectory.append(f"GH_{gh_id}_ID_{id_val}")
                else:
                    # Fallback to just geohash if id is missing
                    daily_trajectory.append(f"GH_{gh_id}")
            else:
                # Just geohash if id is not in dwell_extra_columns
                daily_trajectory.append(f"GH_{gh_id}")

            # --- 4. Temporal/Functional/Flow Context ---
            d_part = getattr(row, "day_part", None)
            if pd.isna(d_part):
                hour = row.start_date_time.hour
                if MORNING_START <= hour < LATE_MORNING_START:
                    d_part = "morning"
                elif LATE_MORNING_START <= hour < AFTERNOON_START:
                    d_part = "late_morning"
                elif AFTERNOON_START <= hour < EVENING_START:
                    d_part = "afternoon"
                elif EVENING_START <= hour < NIGHT_START:
                    d_part = "evening"
                else:
                    d_part = "night"

            if pd.notna(d_part):
                daily_trajectory.append(f"PART_{d_part.upper()}")

            d_type = getattr(row, "day_type", None)
            if pd.notna(d_type):
                daily_trajectory.append(f"DAY_{d_type.upper()}")

            dur = getattr(row, "duration_hours", 0)
            if pd.notna(dur):
                if dur < DUR_SHORT_THRESH:
                    daily_trajectory.append("DUR_SHORT")
                elif dur < DUR_MEDIUM_THRESH:
                    daily_trajectory.append("DUR_MEDIUM")
                else:
                    daily_trajectory.append("DUR_LONG")
                

            if getattr(row, "flag_home", False):
                daily_trajectory.append("TYPE_HOME")
            if getattr(row, "flag_work", False):
                daily_trajectory.append("TYPE_WORK")

            # --- 4.5 Extra Dwells Columns as Tags ---
            if dwell_extra_columns:
                # Exclude 'id' from extra tags since it's already combined with geohash
                extra_cols = [col for col in dwell_extra_columns if col != "id"]
                if extra_cols:
                    extra_tags = get_tags(row, extra_cols, prefix="DWELL_")
                    daily_trajectory.extend(extra_tags)

            # --- 5. Static Attributes Injection ---
            if gh_id in static_tags_map:
                daily_trajectory.extend(static_tags_map[gh_id])

        # Only add if there were actually dwells
        if len(daily_trajectory) > len(u_gen_tags):
            corpus.append(daily_trajectory)

    print(f"Corpus built. Contains {len(corpus)} trajectories.")
    return corpus


def train_place2vec(
    df_dwells,
    df_static=None,
    poi_columns=None,
    str_type_columns=None,
    dwell_extra_columns=None,
    df_user=None,
    user_cols_general=None,
    user_cols_important=None,
    vector_size=128,
    window=15,
    min_count=2,
    epochs=10,
    workers=4,
    seed=42,
):
    """
    Main function to process data and train the Word2Vec model.
    """
    # 1. Process Static Data
    static_map = process_static_attributes(
        df_static, poi_columns, str_type_columns=str_type_columns
    )

    # 2. Build Corpus
    corpus = build_corpus(
        df_dwells,
        static_map,
        df_user=df_user,
        user_cols_general=user_cols_general,
        user_cols_important=user_cols_important,
        dwell_extra_columns=dwell_extra_columns,
    )

    # 3. Train Model
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-Gram
        workers=workers,
        epochs=epochs,
        seed=seed,
        sample=0,
    )

    print("Model training complete.")
    return model


def save_similarity_network(
    model, table_name, dataset, project=PROJECT, top_k=50
):
    """
    Exports the semantic network: For every Geohash, find its top K most similar peers.
    Saves to a BigQuery table using vectorized operations.
    """
    # 1. Get all Geohash keys and their indices
    gh_keys = [w for w in model.wv.index_to_key if w.startswith("GH_")]
    if not gh_keys:
        print("No geohashes found in the model.")
        return

    print(f"Found {len(gh_keys)} geohashes. Calculating vectorized similarities...")

    # Extract normalized vectors only for geohashes
    indices = [model.wv.key_to_index[k] for k in gh_keys]
    vectors = model.wv.get_normed_vectors()[indices]

    # Matrix multiplication for cosine similarity (N, D) @ (D, N) -> (N, N)
    sim_matrix = np.dot(vectors, vectors.T)

    # Number of neighbors to find (plus one for self-similarity)
    n_gh = len(gh_keys)
    k_total = min(top_k + 1, n_gh)

    # 2. Extract Top-K indices and scores using vectorized numpy operations
    partitioned_indices = np.argpartition(-sim_matrix, k_total - 1, axis=1)[:, :k_total]

    # Get actual scores for these partitioned indices
    row_idx = np.arange(n_gh)[:, None]
    partitioned_scores = sim_matrix[row_idx, partitioned_indices]

    # Sort only the top K+1 items within each row
    sorted_within_top = np.argsort(-partitioned_scores, axis=1)
    top_indices = np.take_along_axis(partitioned_indices, sorted_within_top, axis=1)
    top_scores = np.take_along_axis(partitioned_scores, sorted_within_top, axis=1)

    # 3. Flatten results for DataFrame construction
    source_ghs = np.repeat(gh_keys, k_total)
    target_ghs = np.array(gh_keys)[top_indices.ravel()]
    scores = top_scores.ravel()

    df = pd.DataFrame(
        {
            "source_geohash": source_ghs,
            "target_geohash": target_ghs,
            "similarity_score": scores.astype(float),
        }
    )

    # Filter out self-similarity (GH_A to GH_A) and keep only top_k
    df = df[df["source_geohash"] != df["target_geohash"]].copy()
    df["rank"] = df.groupby("source_geohash").cumcount() + 1
    df = df[df["rank"] <= top_k]

    # 4. Upload to BigQuery
    utils.upload_table_to_bq(df, dataset, table_name, project=project,  if_table_exist="replace")
    print(f"Done! Saved network to BigQuery table: {dataset}.{table_name}")


def _token_type_for_projector(word):
    """Derive a label for a place2vec token when category_map is not provided."""
    if word.startswith("GH_"):
        return "Geohash"
    if word.startswith("HOME"):
        return "Home"
    if word.startswith("WORK"):
        return "Work"
    if word.startswith("POI_") or word.startswith("CAT_"):
        return "POI"
    if word.startswith("TYPE_"):
        return "Function"
    if word.startswith("PART_"):
        return "TimeOfDay"
    if word.startswith("DAY_"):
        return "Day"
    if word.startswith("DUR_"):
        return "Duration"
    if word.startswith("FLOW_"):
        return "Flow"
    if word.startswith("CTX_"):
        return "Context"
    if word.startswith("ATTR_"):
        return "Attribute"
    if word.startswith("CENSUS_"):
        return "Census"
    return "Other"


def export_for_projector(
    model,
    output_dir="projector_files",
    metadata_file="metadata.tsv",
    vector_file="vectors.tsv",
    category_map=None,
    include_geohashes=True,
):
    """
    Export word2vec/place2vec vectors and metadata for t-SNE visualization
    (e.g. TensorFlow Projector: https://projector.tensorflow.org/).

    Parameters
    ----------
    model : gensim.models.Word2Vec or gensim.models.keyedvectors.KeyedVectors
        Trained model; if Word2Vec, uses model.wv.
    output_dir : str
        Directory to write TSV files. Created if missing.
    metadata_file : str
        Filename for metadata TSV (labels for each vector).
    vector_file : str
        Filename for vectors TSV (one row per token, tab-separated dimensions).
    category_map : dict or None
        Optional mapping token -> category label. If None, category is derived
        from token prefix (Geohash, Function, TimeOfDay, etc.).
    include_geohashes : bool
        If True, include GH_* tokens in the export. If False, only export
        tokens that appear in category_map (or all non-GH tokens when
        category_map is None).

    Returns
    -------
    int
        Number of items exported.
    """
    wv = model.wv if hasattr(model, "wv") else model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    meta_path = os.path.join(output_dir, metadata_file)
    vec_path = os.path.join(output_dir, vector_file)

    with io.open(vec_path, "w", encoding="utf-8") as out_v, io.open(
        meta_path, "w", encoding="utf-8"
    ) as out_m:
        out_m.write("Name\tCategory\n")
        count = 0
        for word in wv.index_to_key:
            if category_map is not None:
                if word in category_map:
                    cat = category_map[word]
                elif word.startswith("GH_") and include_geohashes:
                    cat = "Geohash"
                else:
                    continue
            else:
                if word.startswith("GH_") and not include_geohashes:
                    continue
                cat = _token_type_for_projector(word)
            vec = wv[word]
            out_v.write("\t".join([str(x) for x in vec]) + "\n")
            out_m.write(f"{word}\t{cat}\n")
            count += 1

    logger.info(
        "Exported %s items to %s (%s, %s). Upload to projector.tensorflow.org.",
        count, output_dir, vector_file, metadata_file,
    )
    return count


def extract_model_features(model, dataset, table_name, project=PROJECT, ANCHORS_MAP=None):
    """
    Extracts semantic scores using Matrix Multiplication.
    Input: Gensim Word2Vec Model
    Output: Pandas DataFrame with geohash and id columns extracted from model tokens.
    
    The model tokens should be in format: GH_{geohash}_ID_{id} or GH_{geohash}
    If id was included in dwell_extra_columns during training, tokens will have the combined format.
    
    Args:
        model: Trained Word2Vec model
        dataset: BigQuery dataset name
        table_name: BigQuery table name
        project: BigQuery project name
        ANCHORS_MAP: Dictionary mapping score column names to model tokens
    """
    if ANCHORS_MAP is None:
        ANCHORS_MAP = {
            # FUNCTION
            "score_home": "TYPE_HOME",
            "score_work": "TYPE_WORK",
            # TIME / RHYTHM
            "score_morning": "PART_MORNING",
            "score_early_afternoon": "PART_EARLY_AFTERNOON",
            "score_late_afternoon": "PART_LATE_AFTERNOON",
            "score_evening": "PART_EVENING",
            "score_night": "PART_NIGHT",
            "score_friday": "DAY_FRIDAY",
            "score_weekday": "DAY_WEEKDAY",
            "score_saturday": "DAY_SATURDAY",
            # DURATION
            "score_stay_short": "DUR_SHORT",
            "score_stay_med": "DUR_MEDIUM",
            "score_stay_long": "DUR_LONG",
        }
    # 1. PREPARE GEOGRAPHIC VECTORS (The Rows)
    # Filter: Keep only valid Geohashes (with or without ID suffix)
    vocab = model.wv.index_to_key
    gh_indices = [i for i, w in enumerate(vocab) if w.startswith("GH_")]
    gh_keys = [vocab[i] for i in gh_indices]
    
    # Parse geohash and id from tokens (format: GH_{geohash}_ID_{id} or GH_{geohash})
    gh_data = []
    for key in gh_keys:
        # Try to match GH_{geohash}_ID_{id} pattern
        match_with_id = re.match(r"GH_(.+?)_ID_(.+)$", key)
        if match_with_id:
            gh_data.append({
                "token": key,
                "geohash": f"GH_{match_with_id.group(1)}",
                "geohash_clean": match_with_id.group(1),
                "id": match_with_id.group(2)
            })
        else:
            # Fallback to just GH_{geohash}
            match_gh_only = re.match(r"GH_(.+)$", key)
            if match_gh_only:
                gh_data.append({
                    "token": key,
                    "geohash": key,
                    "geohash_clean": match_gh_only.group(1),
                    "id": None
                })
    
    print(f"DEBUG: Found {len(gh_data)} geohash tokens")
    print(f"DEBUG: Tokens with ID: {sum(1 for g in gh_data if g['id'] is not None)}")
    print(f"DEBUG: Sample tokens: {[g['token'] for g in gh_data[:5]]}")
    print(f"DEBUG: Sample tokens with full format: {gh_keys[:10]}")
    
    # Check if any tokens match the ID pattern
    id_pattern_matches = [key for key in gh_keys if re.match(r"GH_.+_ID_.+$", key)]
    print(f"DEBUG: Tokens matching ID pattern: {len(id_pattern_matches)}")
    if id_pattern_matches:
        print(f"DEBUG: Example ID pattern matches: {id_pattern_matches[:3]}")
    else:
        print(f"DEBUG: No tokens found with _ID_ pattern. Model may have been trained without id in dwell_extra_columns.")

    # Get the vectors and NORMALIZE them (crucial for Cosine Similarity)
    # norms = vectors / magnitude
    # accessing .vectors directly avoids overhead
    vectors_gh = model.wv.vectors[gh_indices]
    norms_gh = np.linalg.norm(vectors_gh, axis=1, keepdims=True)
    vectors_gh_norm = vectors_gh / norms_gh
    
    # Create mapping from token to parsed data
    token_to_data = {g["token"]: g for g in gh_data}

    # 2. PREPARE ANCHOR VECTORS (The Columns)
    valid_anchor_keys = []
    valid_anchor_cols = []
    anchor_indices = []

    for col_name, tag in ANCHORS_MAP.items():
        if tag in model.wv.key_to_index:
            valid_anchor_keys.append(tag)
            valid_anchor_cols.append(col_name)
            anchor_indices.append(model.wv.key_to_index[tag])
        else:
            print(f"Warning: Anchor '{tag}' not found in model")

    if not anchor_indices:
        return pd.DataFrame()

    vectors_anchors = model.wv.vectors[anchor_indices]
    norms_anchors = np.linalg.norm(vectors_anchors, axis=1, keepdims=True)
    vectors_anchors_norm = vectors_anchors / norms_anchors

    # 3. THE MAGIC STEP: MATRIX MULTIPLICATION
    # (N_Geohashes x D) dot (D x N_Anchors) = (N_Geohashes x N_Anchors)
    # Result is the Cosine Similarity Matrix (-1 to 1)
    similarity_matrix = np.dot(vectors_gh_norm, vectors_anchors_norm.T)

    # 4. CONVERT TO DATAFRAME
    df = pd.DataFrame(similarity_matrix, columns=valid_anchor_cols)
    
    # Extract geohash and id from parsed token data
    df["geohash"] = [g["geohash"] for g in gh_data]  # Full token with "GH_" prefix
    df["id"] = [g["id"] for g in gh_data]  # Extracted id (None if not in token)
    
    print(f"DEBUG: Total geohash tokens in model: {len(df)}")
    print(f"DEBUG: Tokens with ID: {df['id'].notna().sum()}")
    print(f"DEBUG: Sample geohash tokens: {df['geohash'].head(3).tolist()}")
    print(f"DEBUG: Sample IDs: {df['id'].head(5).tolist()}")
    
    # Reorder columns to put identifiers first
    cols = ["geohash", "id"] + valid_anchor_cols
    # Only include columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    print(f"DEBUG: Final DataFrame columns: {df.columns.tolist()}")
    print(f"DEBUG: Final DataFrame shape: {df.shape}")
    print(f"DEBUG: Final sample data:\n{df.head(3)}")

    utils.upload_table_to_bq(df, dataset, table_name, project=project, if_table_exist="replace")


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics / EDA
# ═══════════════════════════════════════════════════════════════════════════════

def visit_distribution_by_period(
    gs_bucket,
    gs_prefix,
    periods=None,
):
    """
    For each period, read the already-exported GCS Parquet shards via Dask
    and compute the distribution of visits per unique location.

    A "location" is defined as:
      - HOME  (flag_home = TRUE)
      - WORK  (flag_work = TRUE)
      - the POI ``poi_id`` for matched third-place dwells

    Parameters
    ----------
    gs_bucket : str
        GCS bucket name (without ``gs://`` prefix).
    gs_prefix : str
        Folder prefix inside the bucket used by ``export_visits_to_gcs``.
    periods : dict or None
        {period_name: [month_str, ...]}. Defaults to the module-level PERIODS.

    Returns
    -------
    pd.DataFrame
        Columns: period, location_type, location_id, visit_count,
                 unique_users, pois_category, keyword
        Sorted by visit_count descending within each period.
    """
    import dask.dataframe as dd

    if periods is None:
        periods = PERIODS

    needed_cols = [
        "identifier",
        "flag_home",
        "flag_work",
        "poi_id",
        "pois_category",
        "keyword",
    ]

    frames = []
    for period_name in periods:
        uri = f"gs://{gs_bucket}/{gs_prefix}/visits_{period_name}/*.parquet"
        logger.info("Reading %s for visit distribution ...", uri)

        ddf = dd.read_parquet(uri, columns=needed_cols)

        # classify location type and id
        ddf["location_type"] = ddf.apply(
            lambda r: "HOME" if r["flag_home"] else ("WORK" if r["flag_work"] else "THIRD_PLACE"),
            axis=1, meta=("location_type", "str"),
        )
        ddf["location_id"] = ddf.apply(
            lambda r: "HOME" if r["flag_home"] else (
                "WORK" if r["flag_work"] else str(r["poi_id"]) if pd.notna(r["poi_id"]) else "UNKNOWN"
            ),
            axis=1, meta=("location_id", "str"),
        )

        # drop unknown (unmatched geohashes with no POI)
        ddf = ddf[ddf["location_id"] != "UNKNOWN"]

        # aggregate: visit_count + metadata per location (Dask-compatible)
        grp = ddf.groupby(["location_type", "location_id"])
        agg_counts = grp.agg(
            visit_count=("identifier", "count"),
            pois_category=("pois_category", "first"),
            keyword=("keyword", "first"),
        ).compute().reset_index()
        # nunique not supported in Dask agg – compute separately
        nunique = grp["identifier"].nunique().compute().reset_index()
        nunique.columns = ["location_type", "location_id", "unique_users"]
        agg = agg_counts.merge(nunique, on=["location_type", "location_id"])
        agg = agg.sort_values("visit_count", ascending=False)

        agg["period"] = period_name
        frames.append(agg)
        logger.info(
            "%s: %d unique locations, %d total visits",
            period_name,
            len(agg),
            agg["visit_count"].sum(),
        )

    result = pd.concat(frames, ignore_index=True)
    return result


def plot_visit_distribution(df_dist, top_n=30):
    """
    Plot visit-count distributions from the output of
    ``visit_distribution_by_period``.

    Creates two figures:
      1. Bar chart of top-N most-visited third-place locations per period.
      2. Histogram (log-scale) of visit counts by location_type per period.

    Parameters
    ----------
    df_dist : pd.DataFrame
        Output of ``visit_distribution_by_period``.
    top_n : int
        Number of top third-place locations to show in the bar chart.
    """
    import matplotlib.pyplot as plt

    periods = df_dist["period"].unique()

    # ── Figure 1: Top-N third-place categories (grouped by pois_category) ──
    fig1, axes1 = plt.subplots(
        1, len(periods), figsize=(7 * len(periods), 6), sharey=True
    )
    if len(periods) == 1:
        axes1 = [axes1]

    for ax, period in zip(axes1, periods):
        tp = df_dist[
            (df_dist["period"] == period) & (df_dist["location_type"] == "THIRD_PLACE")
        ].copy()
        tp["category"] = tp["pois_category"].fillna("unknown")
        cat_agg = (
            tp.groupby("category")
            .agg(visit_count=("visit_count", "sum"), unique_users=("unique_users", "sum"))
            .reset_index()
            .nlargest(top_n, "visit_count")
        )
        ax.barh(range(len(cat_agg)), cat_agg["visit_count"].values)
        ax.set_yticks(range(len(cat_agg)))
        ax.set_yticklabels(cat_agg["category"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Visit count")
        ax.set_title(f"{period} – top {top_n} categories")

    fig1.tight_layout()

    # ── Figure 2: Histogram of visit counts by type ──
    fig2, axes2 = plt.subplots(
        1, len(periods), figsize=(6 * len(periods), 5), sharey=True
    )
    if len(periods) == 1:
        axes2 = [axes2]

    for ax, period in zip(axes2, periods):
        sub = df_dist[(df_dist["period"] == period) & (df_dist["location_id"] != "UNKNOWN")]
        for ltype in ["HOME", "WORK", "THIRD_PLACE"]:
            vals = sub.loc[sub["location_type"] == ltype, "visit_count"]
            if vals.empty:
                continue
            ax.hist(vals, bins=50, alpha=0.6, label=ltype, log=True)
        ax.set_xlabel("Visits per location")
        ax.set_ylabel("# locations (log)")
        ax.set_title(period)
        ax.legend()

    fig2.tight_layout()
    plt.show()

    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 : Export to GCS  +  Streaming tokenization
# ═══════════════════════════════════════════════════════════════════════════════

def export_visits_to_gcs(
    period_name,
    month_list,
    gs_bucket,
    gs_prefix,
    project=PROJECT,
    table_suffix="_v2",
    if_exists="skip",
):
    """
    Export visits for a period to GCS as Parquet shards.

    Left-joins dwells_monthly_fltrd with dwells_pois_home_work{table_suffix} so that
    unmatched dwells (no home / work / POI) are included with NULL POI
    columns – they will be tokenised as bare geohashes.

    Use table_suffix="_v2" to match the table created by create_dwells_pois_home_work()
    (default). Pass "" for the original dwells_pois_home_work table.

    Parameters
    ----------
    if_exists : str
        "skip" (default) — skip export if parquet files already exist on GCS.
        "replace" — delete existing files and re-export.
    """
    uri = f"gs://{gs_bucket}/{gs_prefix}/visits_{period_name}/*.parquet"

    # check if parquet files already exist on GCS
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    existing = fs.glob(f"{gs_bucket}/{gs_prefix}/visits_{period_name}/*.parquet")
    if existing:
        if if_exists == "skip":
            logger.info(
                "SKIP export for %s — %d parquet files already exist at gs://%s/%s/visits_%s/",
                period_name, len(existing), gs_bucket, gs_prefix, period_name,
            )
            return uri
        elif if_exists == "replace":
            logger.info("Deleting %d existing parquet files for %s ...", len(existing), period_name)
            for f_path in existing:
                fs.rm(f_path)
        else:
            raise ValueError(f"if_exists must be 'skip' or 'replace', got '{if_exists}'")

    month_csv = ", ".join(f"'{m}'" for m in month_list)

    query = f"""
    EXPORT DATA OPTIONS(
      uri='{uri}',
      format='PARQUET',
      overwrite=true
    ) AS

    WITH base AS (
      SELECT
        d.identifier,
        d.timestamp,
        d.duration_hours,
        d.start_date_time,
        TIMESTAMP_MICROS(CAST(d.end_date_time / 1000 AS INT64)) AS end_date_time,
        d.date,
        d.geohash AS dwell_geohash,
        DATE_TRUNC(d.date, MONTH) AS month,
        EXTRACT(DAYOFWEEK FROM d.date) AS dow,
        EXTRACT(HOUR FROM d.start_date_time) AS hour
      FROM `{TABLE_DWELLS_RAW}` d
      WHERE DATE_TRUNC(d.date, MONTH) IN ({month_csv})
    ),

    enriched AS (
      SELECT
        b.*,
        p.flag_home,
        p.flag_work,
        p.id          AS poi_id,
        p.pois_category,
        p.keyword,
        p.poi_source,
        p.dist_to_home,
        p.dist_to_work,
        p.dist_to_poi,
        p.day_type,
        p.day_order,
        p.distance_from_prev,
        p.prev_poi_id,
        p.prev_flag_home,
        p.prev_flag_work,
        p.time_diff_from_prev,
        p.census_id,
        p.context_type
      FROM base b
      LEFT JOIN `{TABLE_DWELLS_POIS}{table_suffix}` p
        ON b.identifier = p.identifier
       AND b.timestamp  = p.timestamp
    )

    SELECT e.*,
      COALESCE(e.census_id, sa.yishuv_stat_2022) AS census_id_filled
    FROM enriched e
    LEFT JOIN `{TABLE_GEOHASH_SA}` sa
      ON e.dwell_geohash = sa.geohash
    """

    logger.info("Exporting %s (%d months) → %s", period_name, len(month_list), uri)
    logger.info("Query:\n%s", query)
    utils.read_query_to_dataframe(query, project=project)
    logger.info("Export complete for %s", period_name)


# ── Tokenization helpers ─────────────────────────────────────────────────────

def _time_period(hour):
    """Map hour → day-part label (same buckets as build_corpus)."""
    MORNING_START = 6
    LATE_MORNING_START = 9
    AFTERNOON_START = 12
    EVENING_START = 17
    NIGHT_START = 23

    if MORNING_START <= hour < LATE_MORNING_START:
        return "MORNING"
    elif LATE_MORNING_START <= hour < AFTERNOON_START:
        return "LATE_MORNING"
    elif AFTERNOON_START <= hour < EVENING_START:
        return "AFTERNOON"
    elif EVENING_START <= hour < NIGHT_START:
        return "EVENING"
    return "NIGHT"



# ── Scheme 5 helpers (TLV flag + POI location suits RW) ──────────────────────

_POI_LOCATION_RW_FLAG_LOOKUP = None  # poi_id -> 0|1


def _tlv_label(census_id):
    """Return TLV if census id starts with 5000 (Tel Aviv–Yafo), else NOT_TLV."""
    s = "" if census_id is None or (isinstance(census_id, float) and pd.isna(census_id)) else str(census_id)
    if s.endswith(".0"):
        s = s[:-2]
    return "TLV" if s.startswith("5000") else "NOT_TLV"


def _tlv_label_series(census_ids):
    """Vectorised TLV / NOT_TLV from census id series."""
    s = census_ids.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return pd.Series(
        np.where(s.str.startswith("5000"), "TLV", "NOT_TLV"),
        index=census_ids.index,
        dtype="object",
    )


def _get_poi_location_rw_flag(project=PROJECT):
    """
    Lazy-load {poi_id_str -> flag_location_suits_rw (0|1)} from
    phd-habidatum.phd_pois.google_pois_features.

    flag_location_suits_rw = 1 when location is NOT a work zone/institute
    (suitable as third place / RW-friendly); 0 means office-like location.
    """
    global _POI_LOCATION_RW_FLAG_LOOKUP
    if _POI_LOCATION_RW_FLAG_LOOKUP is not None:
        return _POI_LOCATION_RW_FLAG_LOOKUP

    query = f"""
    SELECT
      CAST(id AS STRING) AS poi_id,
      CAST(flag_location_suits_rw AS INT64) AS flag_location_suits_rw
    FROM `{TABLE_GOOGLE_POIS_FEATURES}`
    WHERE flag_location_suits_rw IS NOT NULL
    """
    logger.info("Loading POI flag_location_suits_rw lookup from BQ ...")
    df = utils.read_query_to_dataframe(query, project=project)
    _POI_LOCATION_RW_FLAG_LOOKUP = {
        str(r["poi_id"]): int(r["flag_location_suits_rw"])
        for _, r in df.iterrows()
        if pd.notna(r["poi_id"]) and str(r["poi_id"]) not in ("", "nan")
    }
    logger.info(
        "Loaded flag_location_suits_rw for %d POIs",
        len(_POI_LOCATION_RW_FLAG_LOOKUP),
    )
    return _POI_LOCATION_RW_FLAG_LOOKUP


def _poi_loc_label(poi_id, lookup=None):
    """
    Map flag_location_suits_rw to token segment.
    0 -> office_loc; 1 or missing -> third_place.
    """
    if lookup is None:
        lookup = _get_poi_location_rw_flag()
    pid = "" if poi_id is None or (isinstance(poi_id, float) and pd.isna(poi_id)) else str(poi_id)
    if pid.endswith(".0"):
        pid = pid[:-2]
    flag = lookup.get(pid)
    return "office_loc" if flag == 0 else "third_place"


def _poi_loc_label_series(poi_ids):
    """Vectorised office_loc / third_place labels for a POI id series."""
    lookup = _get_poi_location_rw_flag()
    s = poi_ids.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    flags = s.map(lookup)  # NaN if missing
    return pd.Series(
        np.where(flags.fillna(1).astype(float) == 0, "office_loc", "third_place"),
        index=poi_ids.index,
        dtype="object",
    )



def _tokenize_row(row, scheme, home_census, work_census):
    """
    Return a list of tokens for one dwell row according to *scheme* (0–5).

    Parameters
    ----------
    row : dict-like  with keys dwell_geohash, flag_home, flag_work,
          poi_id, pois_category, census_id_filled, context_type, hour
    scheme : int  (0, 1, 2, 3, 4, 5)
    home_census, work_census : str or None
    """
    tokens = []
    gh = row.get("dwell_geohash", "")
    flag_home = row.get("flag_home") or False
    flag_work = row.get("flag_work") or False
    poi_id = row.get("poi_id")
    category = row.get("pois_category")
    census = row.get("census_id_filled")
    context = row.get("context_type")
    hour = row.get("hour", 0)
    period = _time_period(hour)

    if flag_home:
        if scheme == 0:
            tokens.append("HOME")
        elif scheme == 5:
            tokens.append(f"HOME_{_tlv_label(home_census)}_WEEKDAY_{period}")
        else:
            tokens.append(f"HOME_{home_census}" if home_census else "HOME")
    elif flag_work:
        if scheme == 0:
            tokens.append("WORK")
        elif scheme == 5:
            tokens.append(f"WORK_{_tlv_label(work_census)}")
        else:
            tokens.append(f"WORK_{work_census}" if work_census else "WORK")
    else:
        # Place token
        if poi_id is not None and str(poi_id) != "" and str(poi_id) != "nan":
            if scheme == 0:
                cat_str = str(category).replace(" ", "_") if category else "UNKNOWN"
                tokens.append(f"CAT_{cat_str}")
            elif scheme == 1:
                tokens.append(f"POI_{poi_id}")
            elif scheme == 2:
                tokens.append(f"POI_{poi_id}")
            elif scheme == 3:
                tokens.append(f"POI_{poi_id}_WEEKDAY_{period}")
            elif scheme == 5:
                cat_str = str(category).replace(" ", "_") if category else "UNKNOWN"
                tokens.append(
                    f"POI_{cat_str}_{_tlv_label(census)}_{_poi_loc_label(poi_id)}"
                    f"_WEEKDAY_{period}"
                )
        else:
            # Unknown / no POI match
            if scheme == 0:
                ctx_str = str(context).replace(" ", "_") if pd.notna(context) else "UNKNOWN"
                tokens.append(f"CTX_{ctx_str}")
            elif scheme == 1:
                tokens.append(f"GH_{gh[:8]}" if len(gh) >= 8 else f"GH_{gh}")
            elif scheme == 2:
                tokens.append(f"CENSUS_{census}" if census else f"GH_{gh[:6]}")
            elif scheme == 3:
                gh8 = gh[:8] if len(gh) >= 8 else gh
                tokens.append(f"GH_{gh8}_{period}")
            elif scheme == 5:
                tokens.append(f"GH_{_tlv_label(census)}_WEEKDAY_{period}")

    return tokens

def _tokenize_df(df, scheme, home_census_col, work_census_col, poi_counts=None):
    """
    Fully vectorised tokenisation of a pandas DataFrame (one token per row).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: flag_home, flag_work, dwell_geohash, poi_id,
        pois_category, context_type, census_id_filled, hour.
    scheme : int  (0–5)
    home_census_col : pd.Series (str)
        Per-row home census ID (aligned with df.index). Empty string = unknown.
    work_census_col : pd.Series (str)
        Per-row work census ID (aligned with df.index). Empty string = unknown.
    poi_counts : set or None
        If provided, only POIs in this set are kept; others become "unknown".

    Returns
    -------
    pd.Series of str  — one token per row ("" if nothing produced).
    """
    tokens = pd.Series("", index=df.index, dtype="object")

    flag_home = df["flag_home"].fillna(False).astype(bool)
    flag_work = df["flag_work"].fillna(False).astype(bool)
    gh = df["dwell_geohash"].fillna("").astype(str)

    poi_id = df["poi_id"].astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    has_poi = (poi_id != "") & (~flag_home) & (~flag_work)

    # filter rare POIs
    if poi_counts is not None:
        has_poi = has_poi & poi_id.isin(poi_counts)

    no_poi = ~flag_home & ~flag_work & ~has_poi

    category = df["pois_category"].fillna("UNKNOWN").astype(str).str.replace(" ", "_")
    context = (
        df["context_type"].astype(str)
        .replace({"nan": "UNKNOWN", "None": "UNKNOWN", "<NA>": "UNKNOWN"})
        .str.replace(" ", "_")
    )
    census = df.get("census_id_filled", pd.Series("", index=df.index)).fillna("").astype(str)
    hour = df["hour"].fillna(0).astype(int)
    period = hour.map(_time_period)

    hc = home_census_col.fillna("").astype(str)
    wc = work_census_col.fillna("").astype(str)

    # ── HOME tokens ──
    if scheme == 0:
        tokens.loc[flag_home] = "HOME"
    elif scheme == 5:
        home_tlv = _tlv_label_series(hc)
        tokens.loc[flag_home] = ("HOME_" + home_tlv + "_WEEKDAY_" + period).loc[flag_home]
    else:
        home_with_c = flag_home & (hc != "")
        home_no_c = flag_home & (hc == "")
        tokens.loc[home_with_c] = "HOME_" + hc.loc[home_with_c]
        tokens.loc[home_no_c] = "HOME"

    # ── WORK tokens ──
    if scheme == 0:
        tokens.loc[flag_work] = "WORK"
    elif scheme == 5:
        work_tlv = _tlv_label_series(wc)
        tokens.loc[flag_work] = ("WORK_" + work_tlv).loc[flag_work]
    else:
        work_with_c = flag_work & (wc != "")
        work_no_c = flag_work & (wc == "")
        tokens.loc[work_with_c] = "WORK_" + wc.loc[work_with_c]
        tokens.loc[work_no_c] = "WORK"

    # ── POI tokens ──
    if scheme == 0:
        tokens.loc[has_poi] = "CAT_" + category.loc[has_poi]
    elif scheme in (1, 2):
        tokens.loc[has_poi] = "POI_" + poi_id.loc[has_poi]
    elif scheme == 3:
        tokens.loc[has_poi] = "POI_" + poi_id.loc[has_poi] + "_WEEKDAY_" + period.loc[has_poi]
    elif scheme == 5:
        poi_tlv = _tlv_label_series(census)
        poi_loc = _poi_loc_label_series(poi_id)
        tokens.loc[has_poi] = (
            "POI_" + category + "_" + poi_tlv + "_" + poi_loc + "_WEEKDAY_" + period
        ).loc[has_poi]

    # ── Unknown / no POI tokens ──
    if scheme == 0:
        tokens.loc[no_poi] = "CTX_" + context.loc[no_poi]
    elif scheme == 1:
        tokens.loc[no_poi] = "GH_" + gh.loc[no_poi].str[:8]
    elif scheme == 2:
        has_census = no_poi & (census != "")
        no_census = no_poi & (census == "")
        tokens.loc[has_census] = "CENSUS_" + census.loc[has_census]
        tokens.loc[no_census] = "GH_" + gh.loc[no_census].str[:6]
    elif scheme == 3:
        tokens.loc[no_poi] = "GH_" + gh.loc[no_poi].str[:8] + "_" + period.loc[no_poi]
    elif scheme == 5:
        gh_tlv = _tlv_label_series(census)
        tokens.loc[no_poi] = ("GH_" + gh_tlv + "_WEEKDAY_" + period).loc[no_poi]

    return tokens


class GCSCorpus:
    """
    Restartable iterable that reads Parquet shards from GCS via Dask,
    applies filters, segments sessions, tokenises, and yields sentences.

    Parameters
    ----------
    gs_uri : str           e.g. "gs://bucket/prefix/visits_pre_covid/*.parquet"
    scheme : int           tokenization scheme 0-5
    home_work_lookup : dict   {(identifier, month): (home_census, work_census)}
    min_duration_hours : float   minimum dwell duration (default 10/60 = 10 min)
    session_gap_hours : float    session break threshold (default 3)
    min_poi_count : int          POI must appear >= this many times (default 5)
    """

    def __init__(
        self,
        gs_uri,
        scheme,
        home_work_lookup,
        min_duration_hours=10 / 60,
        session_gap_hours=3,
        min_poi_count=5,
    ):
        self.gs_uri = gs_uri
        self.scheme = scheme
        self.home_work_lookup = home_work_lookup
        self.min_duration_hours = min_duration_hours
        self.session_gap_hours = session_gap_hours
        self.min_poi_count = min_poi_count
        self._poi_counts = None  # lazy

    # ── first pass: count POI occurrences ──────────────────────────────────
    def _compute_poi_counts(self):
        import dask.dataframe as dd

        logger.info("Computing POI frequency counts (first pass) ...")
        ddf = dd.read_parquet(self.gs_uri, columns=["poi_id"])
        counts = (
            ddf["poi_id"]
            .dropna()
            .value_counts()
            .compute()
        )
        self._poi_counts = set(counts[counts >= self.min_poi_count].index)
        logger.info("Kept %d POIs with >= %d visits", len(self._poi_counts), self.min_poi_count)

    # ── iterator ───────────────────────────────────────────────────────────
    def __iter__(self):
        import dask.dataframe as dd

        if self._poi_counts is None:
            self._compute_poi_counts()

        needed_cols = [
            "identifier", "dwell_geohash", "duration_hours",
            "timestamp", "date", "month",
            "hour", "day_type", "day_order", "time_diff_from_prev",
            "flag_home", "flag_work",
            "poi_id", "pois_category", "census_id_filled", "context_type",
        ]

        ddf = dd.read_parquet(self.gs_uri, columns=needed_cols)

        # duration filter (can be applied in Dask before compute)
        ddf = ddf[ddf["duration_hours"] >= self.min_duration_hours]

        # fast filter: use pre-computed day_type to drop Friday / Saturday
        ddf = ddf[ddf["day_type"] == "weekday"]

        # build Israel non-weekday calendar (holidays, erev chag, elections)
        # for the remaining Sun-Thu dates that still fall on special days
        sample_dates = ddf["date"].head(1000)
        years_in_data = sorted(
            set(pd.to_datetime(sample_dates).dt.year.dropna().unique())
            | set(range(2019, 2025))  # safety fallback
        )
        non_weekday_dates = _build_israel_non_weekdays(years_in_data)

        # process per partition (each is a pandas DF)
        n_partitions = ddf.npartitions
        for p_idx, partition in enumerate(ddf.to_delayed()):
            pdf = partition.compute()
            if pdf.empty:
                continue

            pdf["date"] = pd.to_datetime(pdf["date"])

            # second pass: exclude holidays, erev chag, elections on Sun-Thu
            pdf = pdf[~pdf["date"].dt.date.isin(non_weekday_dates)]
            if pdf.empty:
                continue
            pdf = pdf.sort_values(["identifier", "day_order"])

            # ── vectorised census lookup via merge ──
            pdf["_month_str"] = pdf["month"].astype(str)
            if not hasattr(self, "_lookup_df"):
                # build lookup DataFrame once (lazy, cached on self)
                rows = [
                    {"identifier": uid, "_month_str": mo, "_home_c": hc, "_work_c": wc}
                    for (uid, mo), (hc, wc) in self.home_work_lookup.items()
                ]
                self._lookup_df = pd.DataFrame(rows) if rows else pd.DataFrame(
                    columns=["identifier", "_month_str", "_home_c", "_work_c"]
                )
            pdf = pdf.merge(self._lookup_df, on=["identifier", "_month_str"], how="left")

            # ── session segmentation (vectorised, no per-user loop) ──
            # new session when: different user, or gap > threshold, or first of day
            diff_user = pdf["identifier"] != pdf["identifier"].shift(1)
            tdiff = pdf["time_diff_from_prev"].fillna(0).abs()
            big_gap = (tdiff > self.session_gap_hours) | pdf["time_diff_from_prev"].isna()
            pdf["_session"] = (diff_user | big_gap).cumsum()

            # ── vectorised tokenisation (entire partition, no loops) ──
            pdf["_token"] = _tokenize_df(
                pdf, self.scheme,
                home_census_col=pdf["_home_c"],
                work_census_col=pdf["_work_c"],
                poi_counts=self._poi_counts,
            )

            # ── extract sentences per session (vectorised groupby → list) ──
            # deduplicate consecutive identical tokens (e.g. HOME HOME HOME → HOME)
            valid = pdf[pdf["_token"] != ""].copy()
            if valid.empty:
                continue
            valid["_prev_token"] = valid.groupby("_session")["_token"].shift(1)
            valid = valid[valid["_token"] != valid["_prev_token"]]
            for _, grp in valid.groupby("_session", sort=False):
                toks = grp["_token"].tolist()
                if toks:
                    yield toks

            if (p_idx + 1) % 5 == 0 or p_idx == n_partitions - 1:
                logger.info("Processed partition %d / %d", p_idx + 1, n_partitions)


def build_home_work_census_lookup(month_list, project=PROJECT):
    """
    Build a lightweight dict  {(identifier, month) -> (home_census, work_census)}
    from user_profiles_monthly (home_yishuv_stat_2022, yishuv_stat_2022).
    """
    month_csv = ", ".join(f"'{m}'" for m in month_list)
    query = f"""
    SELECT
      identifier,
      CAST(month AS STRING) AS month,
      home_yishuv_stat_2022 AS home_census,
      yishuv_stat_2022      AS work_census
    FROM `{TABLE_USER_PROFILES}`
    WHERE month IN ({month_csv})
    """
    logger.info("Query:\n%s", query)
    df = utils.read_query_to_dataframe(query, project=project)
    lookup = {}
    for _, r in df.iterrows():
        lookup[(r["identifier"], r["month"])] = (
            str(r["home_census"]) if pd.notna(r["home_census"]) else None,
            str(r["work_census"]) if pd.notna(r["work_census"]) else None,
        )
    logger.info("Home/work census lookup: %d entries", len(lookup))
    return lookup


def write_corpus_to_gcs(corpus_iterable, gs_uri, buffer_size=5000, if_exists="skip"):
    """
    Materialise a corpus iterable to a text file on GCS
    (one sentence per line, space-separated tokens).

    Writes in buffered chunks for better I/O performance.

    Parameters
    ----------
    if_exists : str
        "skip" (default) — skip writing if the file already exists on GCS.
        "replace" — overwrite the existing file.
    """
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    if fs.exists(gs_uri) and if_exists == "skip":
        logger.info("SKIP write_corpus — file already exists: %s", gs_uri)
        return
    n = 0
    buf = []
    with fs.open(gs_uri, "w") as f:
        for sentence in corpus_iterable:
            buf.append(" ".join(sentence))
            n += 1
            if len(buf) >= buffer_size:
                f.write("\n".join(buf) + "\n")
                buf.clear()
                if n % 50_000 == 0:
                    logger.info("  ... %d sentences written so far", n)
        if buf:
            f.write("\n".join(buf) + "\n")
    logger.info("Wrote %d sentences → %s", n, gs_uri)


class GCSLineSentences:
    """Restartable iterable that reads a text corpus from GCS (one sentence per line)."""

    def __init__(self, gs_uri):
        self.gs_uri = gs_uri

    def __iter__(self):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(self.gs_uri, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 : Training  +  Procrustes Alignment
# ═══════════════════════════════════════════════════════════════════════════════

def train_from_corpus(
    corpus,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,
    negative=15,
    epochs=10,
    workers=1,
    seed=42,
):
    """
    Train a Word2Vec model from any restartable iterable of sentences.

    workers=1 avoids a gensim deadlock when the corpus is a GCS stream
    (job queue never reports epoch progress with workers>1).

    Parameters
    ----------
    corpus : iterable of list[str]
        Can be a GCSCorpus, GCSLineSentences, or plain list.
    """
    logger.info("Training Word2Vec (dim=%d, win=%d, mc=%d, sg=%d, neg=%d) ...",
                vector_size, window, min_count, sg, negative)
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        workers=workers,
        epochs=epochs,
        seed=seed,
        sample=0,
    )
    logger.info("Training complete – vocab size %d", len(model.wv))
    return model


def resolve_anchor_categories(
    anchor_categories,
    scheme,
    model_source,
    model_target=None,
    project=PROJECT,
):
    """
    Convert human-readable category names (e.g. ["hospital", "library"]) into
    the concrete token strings used by the given tokenization *scheme*.

    Scheme 0  →  CAT_<category>  (directly from category name)
    Scheme 1-2 →  POI_<id>      (look up POI ids belonging to each category in BQ)
    Scheme 3  →  POI_<id>_WEEKDAY_<period>  (expand across all day-parts)

    Only tokens present in *model_source* (and *model_target* if given) are returned.

    Parameters
    ----------
    anchor_categories : list[str]
        Category names, e.g. ["hospital", "library"]. Matched case-insensitively
        against `primarytype` (Google) and `pois_category` (OSM).
    scheme : int  (0, 1, 2, 3)
    model_source : gensim Word2Vec
    model_target : gensim Word2Vec or None
    project : str

    Returns
    -------
    list[str]   anchor tokens present in the model(s).
    """
    from google.cloud import bigquery

    if scheme == 0:
        # Scheme 0 uses CAT_<category> tokens directly
        candidates = [f"CAT_{c.replace(' ', '_')}" for c in anchor_categories]
    else:
        # Schemes 1-3: resolve categories → POI ids via BQ
        cat_lower = [c.lower() for c in anchor_categories]
        cat_sql = ", ".join(f"'{c}'" for c in cat_lower)
        query = f"""
            SELECT DISTINCT CAST(id AS STRING) AS poi_id
            FROM `{TABLE_GOOGLE_POIS}`
            WHERE LOWER(primarytype) IN ({cat_sql})
            UNION DISTINCT
            SELECT DISTINCT CAST(id AS STRING) AS poi_id
            FROM `{TABLE_OSM_POIS}`
            WHERE LOWER(pois_category) IN ({cat_sql})
        """
        logger.info("Resolving anchor categories %s via BQ ...", anchor_categories)
        logger.info("Query:\n%s", query)
        client = bigquery.Client(project=project)
        df = client.query(query).to_dataframe()
        poi_ids = df["poi_id"].tolist()
        logger.info("Found %d POI ids for categories %s", len(poi_ids), anchor_categories)

        if scheme in (1, 2):
            candidates = [f"POI_{pid}" for pid in poi_ids]
        elif scheme == 3:
            periods = ["MORNING", "LATE_MORNING", "AFTERNOON", "EVENING", "NIGHT"]
            candidates = [f"POI_{pid}_WEEKDAY_{p}" for pid in poi_ids for p in periods]
        else:
            candidates = [f"POI_{pid}" for pid in poi_ids]

    # keep only tokens that exist in the model(s)
    vocab_s = set(model_source.wv.key_to_index)
    present = [t for t in candidates if t in vocab_s]
    if model_target is not None:
        vocab_t = set(model_target.wv.key_to_index)
        present = [t for t in present if t in vocab_t]

    logger.info(
        "Anchor categories %s → %d candidate tokens → %d present in model(s)",
        anchor_categories, len(candidates), len(present),
    )
    return sorted(present)


def align_model_procrustes(
    model_source,
    model_target,
    anchor_tokens=None,
    anchor_categories=None,
    scheme=None,
    min_anchor=20,
):
    """
    Align *model_target* onto *model_source* using Orthogonal Procrustes.

    Parameters
    ----------
    model_source : gensim Word2Vec   (reference space, e.g. pre-COVID)
    model_target : gensim Word2Vec   (to be rotated, e.g. post-COVID)
    anchor_tokens : list[str] or None
        Explicit token list. If None, falls back to *anchor_categories* or
        auto-detected HOME_*/WORK_* tokens.
    anchor_categories : list[str] or None
        Human-readable category names (e.g. ["hospital", "library"]).
        Resolved to tokens via ``resolve_anchor_categories()``.
        Requires *scheme* to be set.
    scheme : int or None
        Required when *anchor_categories* is provided.
    min_anchor : int
        Minimum number of shared anchors required.

    Returns
    -------
    model_target_aligned : gensim Word2Vec
        A *copy* of model_target with rotated vectors.
    R : ndarray
        The rotation matrix.
    """
    wv_s = model_source.wv
    wv_t = model_target.wv

    if anchor_tokens is None and anchor_categories is not None:
        if scheme is None:
            raise ValueError("scheme must be provided when using anchor_categories")
        anchor_tokens = resolve_anchor_categories(
            anchor_categories, scheme, model_source, model_target,
        )
        logger.info("Resolved %d anchor tokens from categories %s",
                     len(anchor_tokens), anchor_categories)

    if anchor_tokens is None:
        shared = set(wv_s.key_to_index) & set(wv_t.key_to_index)
        anchor_tokens = sorted(
            w for w in shared
            if w.startswith("HOME_") or w.startswith("WORK_")
        )
        logger.info("Auto-detected %d anchor tokens (HOME_*/WORK_*)", len(anchor_tokens))

    # filter to tokens present in both
    anchors = [t for t in anchor_tokens if t in wv_s.key_to_index and t in wv_t.key_to_index]
    if len(anchors) < min_anchor:
        raise ValueError(
            f"Only {len(anchors)} shared anchors (need >= {min_anchor}). "
            "Provide more anchor_tokens or lower min_anchor."
        )

    A = np.array([wv_s[t] for t in anchors])  # source
    B = np.array([wv_t[t] for t in anchors])  # target

    # Normalize rows to unit length (standard for embedding alignment)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)

    R, _ = orthogonal_procrustes(B, A)  # find R s.t. B @ R ≈ A

    # apply rotation to ALL target vectors
    import copy
    model_aligned = copy.deepcopy(model_target)
    model_aligned.wv.vectors = model_aligned.wv.vectors @ R

    # also rotate the syn1neg (output weights) if they exist
    if hasattr(model_aligned, "syn1neg"):
        model_aligned.syn1neg = model_aligned.syn1neg @ R

    logger.info("Procrustes alignment done (%d anchors, residual=%.6f)",
                len(anchors), np.linalg.norm(B @ R - A))
    return model_aligned, R


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 : Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_stability(model, category_token_map, top_k=5):
    """
    Stability check: for "non-third-place" categories, what share of
    a token's top-K neighbours belong to the *same* category?

    Parameters
    ----------
    model : gensim Word2Vec
    category_token_map : dict   {category_name: [token1, token2, ...]}
        e.g. {"Gas": ["POI_abc", "POI_def"], "School": [...]}
    top_k : int

    Returns
    -------
    pd.DataFrame  with columns [category, token, matching_rate]
    """
    rows = []
    wv = model.wv
    for cat, tokens in category_token_map.items():
        for tok in tokens:
            if tok not in wv.key_to_index:
                continue
            neighbours = wv.most_similar(tok, topn=top_k)
            neighbour_tokens = {n[0] for n in neighbours}
            same_cat = neighbour_tokens & set(tokens)
            rate = len(same_cat) / top_k
            rows.append({"category": cat, "token": tok, "matching_rate": rate})
    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("category")["matching_rate"].mean()
        logger.info("Stability check:\n%s", summary.to_string())
    return df


def validate_remote_work_rank(
    model,
    ground_truth_poi_ids,
    work_anchor="WORK",
    work_prefix="WORK_",
):
    """
    For each ground-truth POI, find its rank in the most_similar list
    to the WORK concept vector.

    Parameters
    ----------
    model : gensim Word2Vec
    ground_truth_poi_ids : list[str]
        POI IDs (will be prefixed with 'POI_').
    work_anchor : str
        Exact token for work (scheme 0).
    work_prefix : str
        Prefix for census-level work tokens (schemes 1-3).

    Returns
    -------
    pd.DataFrame  with columns [poi_id, rank]  (rank=None if not found)
    """
    wv = model.wv

    # build work vector: either single token or mean of WORK_* tokens
    if work_anchor in wv.key_to_index:
        work_vec = wv[work_anchor]
    else:
        work_tokens = [t for t in wv.key_to_index if t.startswith(work_prefix)]
        if not work_tokens:
            logger.warning("No work tokens found in model!")
            return pd.DataFrame()
        work_vec = np.mean([wv[t] for t in work_tokens], axis=0)

    # get full similarity ranking
    all_sims = wv.similar_by_vector(work_vec, topn=len(wv))
    rank_map = {tok: i + 1 for i, (tok, _) in enumerate(all_sims)}

    rows = []
    for pid in ground_truth_poi_ids:
        tok = f"POI_{pid}"
        rank = rank_map.get(tok)
        rows.append({"poi_id": pid, "token": tok, "rank": rank})

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["rank"])
    if not valid.empty:
        logger.info(
            "Remote-work rank test: median rank = %.0f  (of %d found / %d total)",
            valid["rank"].median(), len(valid), len(df),
        )
    return df


def build_third_place_centroid(model, leisure_tokens, normalize=True):
    """
    Build a synthetic THIRD_PLACE_CENTROID vector from a curated list of
    pure leisure/social tokens (e.g. evening bars, weekend parks, cinemas,
    community centers) by averaging their embeddings in the model.

    Use this centroid as a conceptual anchor in the vector space to measure
    how much a place "acts like" a Third Place (e.g. via cosine similarity).

    Parameters
    ----------
    model : gensim.models.Word2Vec
        Trained place2vec model (typically pre_covid) whose vocabulary
        contains the leisure tokens.
    leisure_tokens : list[str]
        Curated list of token strings that represent pure leisure/social
        places, e.g. ["CAT_bar", "CAT_park", "CAT_cinema", "CAT_community_center"]
        or POI-based tokens like "POI_<id>" depending on tokenization scheme.
    normalize : bool, default True
        If True, return a unit-norm vector so that dot product equals
        cosine similarity when comparing to other normalized vectors.

    Returns
    -------
    centroid : np.ndarray
        Shape (vector_size,) — the Third Place reference centroid.
    tokens_used : list[str]
        Subset of leisure_tokens that were present in the model (others
        are skipped with a warning).

    Examples
    --------
    >>> leisure = ["CAT_bar", "CAT_park", "CAT_cinema", "CAT_community_center"]
    >>> centroid, used = build_third_place_centroid(pre_covid_model, leisure)
    >>> sims = pre_covid_model.wv.similar_by_vector(centroid, topn=20)
    """
    wv = model.wv
    present = [t for t in leisure_tokens if t in wv.key_to_index]
    missing = set(leisure_tokens) - set(present)
    if missing:
        logger.warning(
            "Third-place centroid: %d token(s) not in model (skipped): %s",
            len(missing), sorted(missing)[:10],
        )
    if not present:
        raise ValueError(
            "None of the leisure_tokens are in the model. "
            "Check token strings and model vocabulary."
        )
    vectors = np.array([wv[t] for t in present], dtype=np.float64)
    centroid = np.mean(vectors, axis=0)
    if normalize:
        n = np.linalg.norm(centroid)
        if n > 0:
            centroid = centroid / n
    return centroid, present


def avg_similarity_to_centroid(
    model,
    tokens,
    centroid,
    min_similarity=0.0,
):
    """
    Average cosine similarity of a list of tokens to a centroid vector.
    Tokens not in the model are skipped. Optionally exclude tokens with
    similarity below a threshold (e.g. negative) so the average is not
    pulled down by anti-correlated tokens.

    Parameters
    ----------
    model : gensim.models.Word2Vec
        Model whose vectors to use.
    tokens : list[str]
        Token strings (e.g. leisure_tokens).
    centroid : np.ndarray
        Reference vector (typically unit-norm from build_third_place_centroid).
    min_similarity : float, default 0.0
        Only include tokens with similarity >= min_similarity in the average.
        Use 0.0 to exclude negative correlations; use -np.inf to include all.

    Returns
    -------
    avg_sim : float
        Mean similarity over tokens that are in the model and pass the
        min_similarity filter. np.nan if no such tokens.
    tokens_used : list[str]
        Tokens that were included in the average (in model and >= min_similarity).
    """
    wv = model.wv
    present = [t for t in tokens if t in wv.key_to_index]
    if not present:
        return np.nan, []
    sims = wv.cosine_similarities(centroid, [wv[t] for t in present])
    if min_similarity > -np.inf:
        kept = [
            (t, s) for t, s in zip(present, sims)
            if s >= min_similarity
        ]
        if not kept:
            return np.nan, []
        tokens_used = [t for t, _ in kept]
        avg_sim = float(np.mean([s for _, s in kept]))
        return avg_sim, tokens_used
    return float(np.mean(sims)), present


# Day-period tokens used for POI WORK vs Third Place comparison (scheme 3)
POI_DAY_PERIODS = _CONFIG["poi_day_periods"]


def build_work_centroid(model, work_anchor="WORK", work_prefix="WORK_", normalize=True):
    """
    Build a synthetic WORK_CENTROID vector from the model's WORK token(s).

    Parameters
    ----------
    model : gensim.models.Word2Vec
        Trained place2vec model.
    work_anchor : str
        Exact token for work in scheme 0, e.g. "WORK".
    work_prefix : str
        Prefix for census-level work tokens in schemes 1–3, e.g. "WORK_".
    normalize : bool, default True
        If True, return a unit-norm vector.

    Returns
    -------
    centroid : np.ndarray
        Shape (vector_size,) — the Work reference centroid.
    tokens_used : list[str]
        Token(s) used (single work_anchor or list of WORK_* tokens).
    """
    wv = model.wv
    if work_anchor in wv.key_to_index:
        tokens_used = [work_anchor]
        centroid = wv[work_anchor].astype(np.float64).copy()
    else:
        tokens_used = [t for t in wv.key_to_index if t.startswith(work_prefix)]
        if not tokens_used:
            raise ValueError(
                "No work token(s) found. Check work_anchor and work_prefix."
            )
        centroid = np.mean([wv[t] for t in tokens_used], axis=0)
    if normalize:
        n = np.linalg.norm(centroid)
        if n > 0:
            centroid = centroid / n
    return centroid, tokens_used


def poi_work_thirdplace_similarities(
    model,
    work_centroid,
    third_place_centroid,
    periods=None,
):
    """
    For every POI with scheme-3 day-period tokens, compute similarity of each
    period token to WORK_CENTROID and to THIRD_PLACE_CENTROID, then add
    difference columns and a flag.

    Expects tokens of the form POI_<id>_WEEKDAY_<period> with period in
    MORNING, LATE_MORNING, AFTERNOON, EVENING.

    Parameters
    ----------
    model : gensim.models.Word2Vec
        Trained place2vec model (scheme 3) with POI_*_WEEKDAY_* tokens.
    work_centroid : np.ndarray
        Unit-norm WORK reference vector (e.g. from build_work_centroid).
    third_place_centroid : np.ndarray
        Unit-norm Third Place reference vector (e.g. from build_third_place_centroid).
    periods : list[str] or None
        Day periods to use. Default MORNING, LATE_MORNING, AFTERNOON, EVENING.

    Returns
    -------
    pd.DataFrame
        One row per POI with columns:
        - poi_id
        - Morning_WORK, LATE_Morning_WORK, AFTERNOON_WORK, EVENING_WORK
        - Morning_ThirdPLACE, LATE_Morning_ThirdPLACE, AFTERNOON_ThirdPLACE, EVENING_ThirdPLACE
        - Morning_WORK_minus_EVENING_WORK, LATE_Morning_WORK_minus_EVENING_WORK
        - Morning_ThirdPLACE_minus_AFTERNOON_ThirdPLACE, LATE_Morning_ThirdPLACE_minus_EVENING_ThirdPLACE
        - work_morning_heavy_and_thirdplace_later : bool
          True when (Morning_WORK - EVENING_WORK) > 0 and (LATE_Morning_WORK - EVENING_WORK) > 0
          and (Morning_ThirdPLACE - AFTERNOON_ThirdPLACE) < 0 and (LATE_Morning_ThirdPLACE - EVENING_ThirdPLACE) < 0.
    """
    periods = periods or POI_DAY_PERIODS
    wv = model.wv

    # Collect POI ids that have at least one WEEKDAY_<period> token
    poi_ids = set()
    for key in wv.key_to_index:
        if not key.startswith("POI_") or "_WEEKDAY_" not in key:
            continue
        tail = key[len("POI_"):]
        poi_id, _ = tail.split("_WEEKDAY_", 1)
        poi_ids.add(poi_id)
    poi_ids = sorted(poi_ids)

    def token(pid, period):
        return f"POI_{pid}_WEEKDAY_{period}"

    def sim_to_work(tok):
        if tok not in wv.key_to_index:
            return np.nan
        return float(wv.cosine_similarities(work_centroid, [wv[tok]])[0])

    def sim_to_third(tok):
        if tok not in wv.key_to_index:
            return np.nan
        return float(wv.cosine_similarities(third_place_centroid, [wv[tok]])[0])

    rows = []
    for pid in poi_ids:
        row = {"poi_id": pid}
        for p in periods:
            t = token(pid, p)
            row[_period_col(p, "WORK")] = sim_to_work(t)
            row[_period_col(p, "ThirdPLACE")] = sim_to_third(t)
        # Difference columns
        row["Morning_WORK_minus_EVENING_WORK"] = _safe_minus(
            row.get(_period_col("MORNING", "WORK")),
            row.get(_period_col("EVENING", "WORK")),
        )
        row["LATE_Morning_WORK_minus_EVENING_WORK"] = _safe_minus(
            row.get(_period_col("LATE_MORNING", "WORK")),
            row.get(_period_col("EVENING", "WORK")),
        )
        row["Morning_ThirdPLACE_minus_AFTERNOON_ThirdPLACE"] = _safe_minus(
            row.get(_period_col("MORNING", "ThirdPLACE")),
            row.get(_period_col("AFTERNOON", "ThirdPLACE")),
        )
        row["LATE_Morning_ThirdPLACE_minus_EVENING_ThirdPLACE"] = _safe_minus(
            row.get(_period_col("LATE_MORNING", "ThirdPLACE")),
            row.get(_period_col("EVENING", "ThirdPLACE")),
        )
        # Flag: first two > 0 and last two < 0
        d1 = row["Morning_WORK_minus_EVENING_WORK"]
        d2 = row["LATE_Morning_WORK_minus_EVENING_WORK"]
        d3 = row["Morning_ThirdPLACE_minus_AFTERNOON_ThirdPLACE"]
        d4 = row["LATE_Morning_ThirdPLACE_minus_EVENING_ThirdPLACE"]
        row["work_morning_heavy_and_thirdplace_later"] = (
            _gt0(d1) and _gt0(d2) and _lt0(d3) and _lt0(d4)
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _period_col(period, suffix):
    """Column name for a period similarity, e.g. LATE_MORNING -> LATE_Morning_ThirdPLACE."""
    if period == "LATE_MORNING":
        return f"LATE_Morning_{suffix}"
    return f"{period.capitalize()}_{suffix}"


def _safe_minus(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a) - float(b)


def _gt0(x):
    return x is not None and not np.isnan(x) and float(x) > 0


def _lt0(x):
    return x is not None and not np.isnan(x) and float(x) < 0


# Daytime periods: morning, late_morning, afternoon (exclude evening and night)
# Names must match _period_col() output: period.capitalize() + "_WORK" -> "Afternoon_WORK"
DAYTIME_WORK_COLS = _CONFIG["daytime_work_columns"]
# Evening/night: work similarity must be < threshold or missing to keep POI
EVENING_NIGHT_WORK_COLS = _CONFIG["evening_night_work_columns"]


def pois_high_work_similarity_daytime(
    model,
    work_centroid,
    third_place_centroid=None,
    threshold=0.8,
    project=PROJECT,
):
    """
    For all POIs with scheme-3 day-period tokens, compute WORK (and optionally
    third-place) similarities, add category from
    phd-habidatum.phd_pois.google_osm_frsq_pois_wth_duplicates, and add a flag
    indicating whether the POI qualifies for remote work (high WORK similarity
    in daytime, low or missing in evening/night).

    Parameters
    ----------
    model : gensim.models.Word2Vec
        Trained place2vec model (scheme 3) with POI_*_WEEKDAY_* tokens.
    work_centroid : np.ndarray
        Unit-norm WORK reference vector (e.g. from build_work_centroid).
    third_place_centroid : np.ndarray or None
        If provided, third-place similarities are computed and included.
        If None, a dummy vector is used and third-place columns are omitted.
    threshold : float, default 0.8
        qualifies_remote_work is True when: max daytime WORK sim > threshold and
        each evening/night WORK sim (if present) is < threshold or missing.
    project : str
        GCP project for BigQuery (POI metadata table).

    Returns
    -------
    pd.DataFrame
        All POIs with columns: poi_id, WORK (and optionally ThirdPLACE) similarity
        columns, max_daytime_work_sim, qualifies_remote_work (bool), and
        pois_category, keyword, census_new_neighborhood_type from BQ.
    """
    if third_place_centroid is None:
        dummy = np.zeros(model.wv.vector_size, dtype=np.float32)
        df = poi_work_thirdplace_similarities(model, work_centroid, dummy)
        df = df.drop(columns=[c for c in df.columns if "ThirdPLACE" in c], errors="ignore")
    else:
        df = poi_work_thirdplace_similarities(model, work_centroid, third_place_centroid)

    daytime_cols = [c for c in DAYTIME_WORK_COLS if c in df.columns]
    if not daytime_cols:
        df = df.assign(max_daytime_work_sim=np.nan, qualifies_remote_work=False)
    else:
        df["max_daytime_work_sim"] = df[daytime_cols].max(axis=1)
        # Flag: qualifies if daytime > threshold AND evening/night < threshold or missing
        high_daytime = df["max_daytime_work_sim"] > threshold
        evening_night_cols = [c for c in EVENING_NIGHT_WORK_COLS if c in df.columns]
        low_or_missing_evening = pd.Series(True, index=df.index)
        for c in evening_night_cols:
            low_or_missing_evening = low_or_missing_evening & (pd.isna(df[c]) | (df[c] < threshold))
        df["qualifies_remote_work"] = high_daytime & low_or_missing_evening

    meta = _get_poi_meta_lookup(project=project)
    df["pois_category"] = df["poi_id"].map(lambda pid: meta.get(str(pid), {}).get("pois_category"))
    df["keyword"] = df["poi_id"].map(lambda pid: meta.get(str(pid), {}).get("keyword"))
    df["census_new_neighborhood_type"] = df["poi_id"].map(
        lambda pid: meta.get(str(pid), {}).get("census_new_neighborhood_type")
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 : Analysis helpers  (Q1 – Q4)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Q1: Trajectory N-grams ────────────────────────────────────────────────────

# Lookups for external metadata used in aggregation dimensions
_NEIGHBORHOOD_TYPE_LOOKUP = None        # census_id -> new_neighborhood_type
_POI_META_LOOKUP = None                 # poi_id -> {new_neighborhood_type, pois_category, geohash_context_type}
_GH_CONTEXT_LOOKUP = None               # geohash -> context_type


def _get_neighborhood_type_lookup(project=PROJECT):
    """
    Lazy-load mapping {census_id_str -> new_neighborhood_type} from BQ table
    phd-habidatum.phd_datasets.sa_neigborhoods_types_new.
    """
    global _NEIGHBORHOOD_TYPE_LOOKUP
    if _NEIGHBORHOOD_TYPE_LOOKUP is not None:
        return _NEIGHBORHOOD_TYPE_LOOKUP

    query = f"""
    SELECT
      CAST(yishuv_stat_2022 AS STRING) AS census_id,
      new_neighborhood_type
    FROM `{TABLE_NEIGHBOURHOOD_TYPES}`
    """
    logger.info("Loading neighborhood type lookup from BQ ...")
    df = utils.read_query_to_dataframe(query, project=project)
    _NEIGHBORHOOD_TYPE_LOOKUP = {
        str(r["census_id"]): str(r["new_neighborhood_type"])
        for _, r in df.iterrows()
        if pd.notna(r["census_id"]) and pd.notna(r["new_neighborhood_type"])
    }
    logger.info("Loaded %d neighborhood types", len(_NEIGHBORHOOD_TYPE_LOOKUP))
    return _NEIGHBORHOOD_TYPE_LOOKUP


def _get_poi_meta_lookup(project=PROJECT):
    """
    Lazy-load mapping
    {poi_id_str -> {new_neighborhood_type, pois_category, geohash_context_type, keyword}}
    from phd_pois.google_osm_frsq_pois_wth_duplicates.
    """
    global _POI_META_LOOKUP
    if _POI_META_LOOKUP is not None:
        return _POI_META_LOOKUP

    query = """
    SELECT
      CAST(id AS STRING) AS poi_id,
      census_new_neighborhood_type,
      pois_category,
      geohash_context_type,
      keyword
    FROM `phd_pois.google_osm_frsq_pois_wth_duplicates`
    """
    logger.info("Loading POI metadata lookup from BQ ...")
    df = utils.read_query_to_dataframe(query, project=project)
    meta = {}
    for _, r in df.iterrows():
        pid = str(r["poi_id"])
        if pd.isna(pid) or pid == "":
            continue
        meta[pid] = {
            "census_new_neighborhood_type": r.get("census_new_neighborhood_type"),
            "pois_category": r.get("pois_category"),
            "geohash_context_type": r.get("geohash_context_type"),
            "keyword": r.get("keyword"),
        }
    _POI_META_LOOKUP = meta
    logger.info("Loaded %d POI metadata rows", len(_POI_META_LOOKUP))
    return _POI_META_LOOKUP


def _get_gh_context_lookup(project=PROJECT):
    """
    Lazy-load mapping {geohash_str -> context_type} from
    phd-habidatum.phd_adm.geohashes_with_sa_gushdan_context.
    """
    global _GH_CONTEXT_LOOKUP
    if _GH_CONTEXT_LOOKUP is not None:
        return _GH_CONTEXT_LOOKUP

    query = f"""
    SELECT
      geohash,
      context_type
    FROM `{TABLE_GEOHASH_CONTEXT}`
    """
    logger.info("Loading geohash context lookup from BQ ...")
    df = utils.read_query_to_dataframe(query, project=project)
    _GH_CONTEXT_LOOKUP = {
        str(r["geohash"]): str(r["context_type"])
        for _, r in df.iterrows()
        if pd.notna(r["geohash"]) and pd.notna(r["context_type"])
    }
    logger.info("Loaded %d geohash context rows", len(_GH_CONTEXT_LOOKUP))
    return _GH_CONTEXT_LOOKUP


def _parse_token_for_dims(tok):
    """
    Parse a raw token into components used by aggregation dimensions.

    Returns a dict with keys:
      - kind: 'HOME', 'WORK', 'POI', 'GH', or other leading keyword
      - census_id: str or None
      - poi_id: str or None
      - geohash: str or None
      - period: str or None   (e.g. 'MORNING', 'LATE_MORNING' for scheme 3)
      - raw: original token
    """
    res = {
        "raw": tok,
        "kind": None,
        "census_id": None,
        "poi_id": None,
        "geohash": None,
        "period": None,
    }

    if tok.startswith("HOME"):
        parts = tok.split("_")
        res["kind"] = "HOME"
        if len(parts) >= 2:
            res["census_id"] = parts[1]
        return res

    if tok.startswith("WORK"):
        parts = tok.split("_")
        res["kind"] = "WORK"
        if len(parts) >= 2:
            res["census_id"] = parts[1]
        return res

    if tok.startswith("POI_"):
        # Handles both POI_<id> and POI_<id>_WEEKDAY_<period>
        res["kind"] = "POI"
        tail = tok[len("POI_") :]  # e.g. "123_456", "42449535_hospital_WEEKDAY_LATE_MORNING"
        poi_id = tail
        period = None
        if "_WEEKDAY_" in tail:
            poi_id, period = tail.split("_WEEKDAY_", 1)
        res["poi_id"] = poi_id
        res["period"] = period
        return res

    if tok.startswith("GH_"):
        res["kind"] = "GH"
        tail = tok[3:]  # e.g. "<gh8>" or "<gh8>_MORNING"
        if "_" in tail:
            geohash, period = tail.split("_", 1)
            res["geohash"] = geohash
            res["period"] = period
        else:
            res["geohash"] = tail
        return res

    # Fallback: use first segment as kind, leave others None
    kind = tok.split("_", 1)[0] if tok else ""
    res["kind"] = kind or "OTHER"
    return res


def _dim_neighborhood_type(parsed, neigh_lookup, poi_meta):
    """Return neighborhood type for HOME/WORK (via census) or POI (via poi_id)."""
    if neigh_lookup is None and poi_meta is None:
        return None

    kind = parsed["kind"]
    if kind in {"HOME", "WORK"} and parsed["census_id"]:
        return neigh_lookup.get(parsed["census_id"]) if neigh_lookup is not None else None

    if kind == "POI" and parsed["poi_id"] and poi_meta is not None:
        meta = poi_meta.get(parsed["poi_id"])
        if meta:
            return meta.get("census_new_neighborhood_type")
    return None


def _sanitize_category_label(value):
    """Normalize category strings for token segments (spaces → underscores)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_") or None


def _dim_category(parsed, poi_meta, poi_category_by_id=None):
    """Return POI category if available.

    Prefer ``poi_category_by_id`` (e.g. Google primarytype) when provided,
    otherwise fall back to ``pois_category`` from the POI metadata table.
    """
    if parsed["kind"] != "POI" or not parsed["poi_id"]:
        return None
    poi_id_str = str(parsed["poi_id"])
    if poi_category_by_id and poi_id_str in poi_category_by_id:
        return _sanitize_category_label(poi_category_by_id[poi_id_str])
    if poi_meta is None:
        return None
    meta = poi_meta.get(parsed["poi_id"]) or poi_meta.get(poi_id_str)
    if meta:
        return _sanitize_category_label(meta.get("pois_category"))
    return None


def _dim_context(parsed, poi_meta, gh_context_lookup):
    """Return context type for POI (via POI table) or GH_ (via geohash table)."""
    kind = parsed["kind"]
    if kind == "POI" and parsed["poi_id"] and poi_meta is not None:
        meta = poi_meta.get(parsed["poi_id"])
        if meta:
            return meta.get("geohash_context_type")
    if kind == "GH" and parsed["geohash"] and gh_context_lookup is not None:
        return gh_context_lookup.get(parsed["geohash"])
    return None


def _poi_status_segment(
    poi_id_str,
    poi_status_by_id=None,
    potential_third_ids=None,
    predicted_third_ids=None,
):
    """
    Resolve a single status segment for a POI id.

    If ``poi_status_by_id`` is provided (1006 four-way labeling), return that
    label or ``non_google`` when the id is missing. Otherwise fall back to the
    legacy two-way potential / predicted flags.
    """
    if poi_status_by_id is not None:
        return poi_status_by_id.get(poi_id_str, "non_google")
    if predicted_third_ids and poi_id_str in predicted_third_ids:
        return "predicted_third_place"
    if potential_third_ids and poi_id_str in potential_third_ids:
        return "potential_third_place"
    return None


def _build_aggregated_token(
    parsed,
    use_neigh,
    use_cat,
    use_ctx,
    neigh_lookup,
    poi_meta,
    gh_context_lookup,
    potential_third_ids=None,
    predicted_third_ids=None,
    poi_status_by_id=None,
    poi_category_by_id=None,
):
    """
    Build final aggregated token from base kind + optional dimensions.

    Rules:
      - Base is one of: HOME, WORK, POI, GH, or the raw token for unknown kinds.
      - Dimensions are appended in fixed order:
          neighborhood_type, then category, then context, then (for POI with
          category aggregation) at most one status segment.
      - Only dimensions whose flag is True and value is not None are appended.
      - Status (only when use_cat and kind==POI):
          * If poi_status_by_id is set: that label, else ``non_google``.
          * Else legacy: predicted_third_place / potential_third_place.
      - Category prefers poi_category_by_id (Google primarytype) when set.
    """
    kind = parsed["kind"]
    raw = parsed["raw"]

    if kind not in {"HOME", "WORK", "POI", "GH"}:
        # Unknown token type: leave unchanged
        return raw

    parts = [kind]

    if use_neigh:
        neigh = _dim_neighborhood_type(parsed, neigh_lookup, poi_meta)
        if neigh:
            parts.append(str(neigh))

    if use_cat and kind == "POI":
        cat = _dim_category(parsed, poi_meta, poi_category_by_id=poi_category_by_id)
        if cat:
            parts.append(str(cat))

    if use_ctx:
        ctx = _dim_context(parsed, poi_meta, gh_context_lookup)
        if ctx:
            parts.append(str(ctx))

    if (use_cat or poi_status_by_id is not None or potential_third_ids or predicted_third_ids) and kind == "POI":
        poi_id = parsed.get("poi_id")
        if poi_id:
            status = _poi_status_segment(
                str(poi_id),
                poi_status_by_id=poi_status_by_id,
                potential_third_ids=potential_third_ids,
                predicted_third_ids=predicted_third_ids,
            )
            if status:
                parts.append(status)

    # Preserve scheme-3 day-part information for POI and GH tokens, when present.
    period = parsed.get("period")
    if period:
        if kind == "POI":
            # Match original pattern POI_<...>_WEEKDAY_<period>
            parts.append(f"WEEKDAY_{period}")
        elif kind == "GH":
            # Original pattern GH_<geohash>_<period>
            parts.append(str(period))

    return "_".join(parts)


def extract_top_ngrams(
    corpus,
    n=3,
    top_k=50,
    exclude_prefixes=None,
    aggregate_by_category=False,
    aggregate_by_neighborhood_type=False,
    aggregate_by_context=False,
    split_by_home_neighborhood=False,
    potential_third_ids=None,
    predicted_third_ids=None,
    poi_status_by_id=None,
    poi_category_by_id=None,
):
    """
    Extract top-K most frequent n-grams from a corpus iterable.

    Parameters
    ----------
    corpus : iterable of list[str]
    n : int   n-gram size (default 3)
    top_k : int   number of results to return (default 50)
    exclude_prefixes : list[str] or None
        If provided, skip n-grams where ALL tokens start with one of these
        prefixes. E.g. ["HOME", "WORK"] to filter out pure home/work combos.
    aggregate_by_category : bool
        If True, for POI tokens add a pois_category segment (from
        phd_pois.google_osm_frsq_pois_wth_duplicates) to the base POI token.
        Overridden per-id by ``poi_category_by_id`` when provided.
    aggregate_by_neighborhood_type : bool
        If True, map:
          - HOME_<census> / WORK_<census> to HOME_<new_neighborhood_type> /
            WORK_<new_neighborhood_type> using sa_neigborhoods_types_new
            (join on census id),
          - POI tokens to POI_<new_neighborhood_type> using the POI metadata
            table.
    aggregate_by_context : bool
        If True:
          - For POI tokens, append geohash_context_type from the POI metadata
            table.
          - For GH_ tokens, append context_type from
            geohashes_with_sa_gushdan_context.
    split_by_home_neighborhood : bool
        If True, ignore n-grams that do not contain any HOME token, and split
        counts by the home neighborhood type derived from the FIRST HOME token
        in the n-gram. The home neighborhood type is taken from the neighbourhood
        dimension (e.g. with aggregate_by_neighborhood_type=True this will be
        the new_neighborhood_type for the user's home).
    potential_third_ids : set[str] or None
        Legacy: POI ids with flag_location_suits_rw = 1. Used only when
        ``poi_status_by_id`` is None.
    predicted_third_ids : set[str] or None
        Legacy: POI ids with prediction_third_place = 1. Used only when
        ``poi_status_by_id`` is None.
    poi_status_by_id : dict[str, str] or None
        Optional map poi_id -> status label for four-way tagging, e.g.
        ``third_place``, ``potential_but_not_third_place``, ``likely_office``.
        Ids missing from the map become ``non_google`` when this dict is set.
    poi_category_by_id : dict[str, str] or None
        Optional map poi_id -> category label (e.g. Google primarytype).
        Preferred over metadata ``pois_category`` when present.

    When multiple flags are True, the final token is built as:
      BASE[_neighborhood_type][_category][_context][_status]
    with only applicable dimensions appended, in that order.

    Returns
    -------
    - If split_by_home_neighborhood is False:
        list of (ngram_tuple, percentage), where percentage is
        100 * count / (total number of n-gram occurrences in the corpus).
    - If split_by_home_neighborhood is True:
        dict mapping home_type -> list of (ngram_tuple, value) (top_k per
        home_type), where value is normalised by the number of unique sentences
        for that home_type.
    """
    use_any_agg = (
        aggregate_by_category
        or aggregate_by_neighborhood_type
        or aggregate_by_context
        or poi_status_by_id is not None
        or poi_category_by_id is not None
    )

    neigh_lookup = None
    poi_meta = None
    gh_context_lookup = None

    # We need neighborhood lookup both for explicit aggregation
    # and for split_by_home_neighborhood (to classify HOME tokens),
    # even if aggregate_by_neighborhood_type is False.
    if aggregate_by_neighborhood_type or split_by_home_neighborhood:
        neigh_lookup = _get_neighborhood_type_lookup()
    if aggregate_by_category or aggregate_by_neighborhood_type or aggregate_by_context:
        # Any dimension that touches POIs needs the POI metadata table
        poi_meta = _get_poi_meta_lookup()
    if aggregate_by_context:
        gh_context_lookup = _get_gh_context_lookup()

    # Global or per-home-type counters
    counter = Counter()
    home_counters = defaultdict(Counter) if split_by_home_neighborhood else None
    sentences_by_home = defaultdict(set) if split_by_home_neighborhood else None

    for sentence in corpus:
        sentence = list(sentence)

        # For split_by_home_neighborhood we also want a per-sentence home_type and
        # to track unique sentences per home_type.
        if split_by_home_neighborhood:
            parsed_sentence = [_parse_token_for_dims(tok) for tok in sentence]
            home_type_sent = None
            for parsed in parsed_sentence:
                if parsed["kind"] == "HOME":
                    home_type_sent = _dim_neighborhood_type(parsed, neigh_lookup, poi_meta)
                    break
            if home_type_sent:
                sentences_by_home[str(home_type_sent)].add(tuple(sentence))

        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            if exclude_prefixes:
                if all(
                    any(tok.startswith(p) for p in exclude_prefixes)
                    for tok in ngram
                ):
                    continue

            parsed_tokens = None
            if use_any_agg or split_by_home_neighborhood:
                parsed_tokens = [_parse_token_for_dims(tok) for tok in ngram]

            if use_any_agg:
                ngram = tuple(
                    _build_aggregated_token(
                        parsed,
                        aggregate_by_neighborhood_type,
                        aggregate_by_category,
                        aggregate_by_context,
                        neigh_lookup,
                        poi_meta,
                        gh_context_lookup,
                        potential_third_ids=potential_third_ids,
                        predicted_third_ids=predicted_third_ids,
                        poi_status_by_id=poi_status_by_id,
                        poi_category_by_id=poi_category_by_id,
                    )
                    for parsed in parsed_tokens
                )

            if split_by_home_neighborhood:
                # Use FIRST HOME token in this n-gram to determine home_type
                if parsed_tokens is None:
                    parsed_tokens = [_parse_token_for_dims(tok) for tok in ngram]
                home_type = None
                for parsed in parsed_tokens:
                    if parsed["kind"] == "HOME":
                        # Derive neighborhood type for this HOME token
                        home_type = _dim_neighborhood_type(parsed, neigh_lookup, poi_meta)
                        break
                if not home_type:
                    # Either no HOME token, or no neighborhood info -> skip n-gram
                    continue
                home_counters[str(home_type)][ngram] += 1
            else:
                counter[ngram] += 1

    if split_by_home_neighborhood:
        # Normalise each n-gram count by the number of unique sentences
        # observed for that home_type.
        result = {}
        for home_type, cnt in home_counters.items():
            total_sent = len(sentences_by_home.get(home_type, ()))
            if total_sent == 0:
                # Fallback: if somehow no sentences tracked, keep absolute counts
                result[home_type] = cnt.most_common(top_k)
            else:
                result[home_type] = [
                    (ng, c / total_sent)
                    for ng, c in cnt.most_common(top_k)
                ]
        return result

    # Global normalisation: absolute count divided by total n-gram occurrences
    total = sum(counter.values())
    if total == 0:
        return []
    return [
        (ngram, 100.0 * count / total)
        for ngram, count in counter.most_common(top_k)
    ]


def top_third_place_ngrams_by_home(
    corpus,
    third_place_categories,
    n=3,
    top_k=20,
    aggregate_by_category=True,
    aggregate_by_neighborhood_type=True,
    aggregate_by_context=False,
):
    """
    For each home neighborhood type, find top-K n-grams that:
      - contain at least one \"third place\" token (from *third_place_categories*),
      - and do NOT contain work or education tokens.

    Parameters
    ----------
    corpus : iterable of list[str]
        Typically a GCSLineSentences or similar, for a given period & scheme.
    third_place_categories : list[str]
        List of category/type strings for third places, e.g. from
        005-google-places.ipynb (bakery, cafe, coffee_shop, ...).
    n : int
        n-gram size (default 3).
    top_k : int
        Number of results to keep per home neighborhood type.
    aggregate_by_* : bool
        Passed through to extract_top_ngrams so that tokens carry
        neighborhood/category/context information as desired.

    Returns
    -------
    dict
        {home_type: {\"top_ngrams\": [(ngram_tuple, value), ...],
                     \"total_prob\": float}}
        where *value* is already normalised by number of unique sentences for
        that home_type (from extract_top_ngrams with split_by_home_neighborhood=True),
        and total_prob is the sum of those values over the filtered n-grams.
    """
    # First, get per-home-type n-grams with normalised values
    base = extract_top_ngrams(
        corpus,
        n=n,
        top_k=1000,  # allow plenty, we'll trim to top_k after filtering
        aggregate_by_category=aggregate_by_category,
        aggregate_by_neighborhood_type=aggregate_by_neighborhood_type,
        aggregate_by_context=aggregate_by_context,
        split_by_home_neighborhood=True,
    )

    third_set = {c.lower() for c in third_place_categories}

    def _is_third_place_token(tok: str) -> bool:
        """Heuristic: check if any third-place category name appears in the token."""
        low = tok.lower()
        return any(tp in low for tp in third_set)

    def _is_work_or_education(tok: str) -> bool:
        low = tok.lower()
        if low.startswith("work"):
            return True
        # education can appear as category or context substring
        return "education" in low

    result = {}
    for home_type, ngrams in base.items():
        filtered = []
        total_prob = 0.0
        for ngram, value in ngrams:
            tokens = list(ngram)
            # Require at least one third-place token
            has_third = any(_is_third_place_token(t) for t in tokens)
            if not has_third:
                continue
            has_forbidden = any(_is_work_or_education(t) for t in tokens)
            if has_forbidden:
                continue
            filtered.append((ngram, value))
            total_prob += value

        # Keep only top_k of the filtered ones
        result[home_type] = {
            "top_ngrams": filtered[:top_k],
            "total_prob": total_prob,
        }

    return result


# ── Q2: Workiness Score ──────────────────────────────────────────────────────

def _poi_token_allowed_day_part(tok: str, allowed_set) -> bool:
    """
    True if token has no day-part (scheme 1/2) or its day-part is in allowed_set.
    allowed_set: set of uppercase names e.g. {"MORNING", "AFTERNOON"}.
    """
    if not allowed_set:
        return True
    low = tok.lower()
    if "weekday_" not in low:
        return True
    if "afternoon" in low:
        return "AFTERNOON" in allowed_set
    if "late_morning" in low:
        return "LATE_MORNING" in allowed_set
    if "morning" in low:
        return "MORNING" in allowed_set
    if "evening" in low:
        return "EVENING" in allowed_set
    if "night" in low:
        return "NIGHT" in allowed_set
    return False


def compute_closeness(
    model,
    poi_tokens=None,
    work_anchor="WORK",
    work_prefix="WORK_",
    aggregate_by_category=False,
    aggregate_by_neighborhood_type=False,
    allowed_day_parts=None,
    min_n_pois=20,
    target="work",
    aggregate_by_context=False,
    include_geohash_tokens=False,
):
    """
    Compute Closeness (workiness / hominess) =
    cosine_sim(location_vector, anchor_vector).

    Parameters
    ----------
    model : gensim Word2Vec
    poi_tokens : list[str] or None
        If None, all POI_* tokens in the model are used (and, optionally,
        GH_* tokens when aggregate_by_context or include_geohash_tokens is True).
    work_anchor, work_prefix : str
        How to define the anchor vector (single token or mean of prefix).
    aggregate_by_category : bool
        If True, group POI tokens by pois_category (from POI metadata) and
        return one row per category with mean workiness and n_pois.
    aggregate_by_neighborhood_type : bool
        If True, group POI tokens by neighborhood type (from POI metadata);
        can be combined with aggregate_by_category for POI_<neigh>_<category>.
    allowed_day_parts : tuple[str] or None
        If set (e.g. ("MORNING", "AFTERNOON")), only include POI tokens whose
        day-part is in this set. Tokens without a day-part (scheme 1/2) are
        always included. Use to exclude NIGHT, EVENING, LATE_MORNING.
    min_n_pois : int
        When aggregating, keep only rows with n_pois / n_tokens >= this
        (default 20).
    target : {"work", "home"}
        Convenience flag for choosing whether to compare locations to WORK
        (default) or HOME. When target="home" and work_anchor / work_prefix
        are left as defaults, the anchor becomes HOME / HOME_.
    aggregate_by_context : bool
        If True, aggregate similarity at the context level (e.g. office,
        residential) using both POI and GH_ tokens. Cannot be combined with
        aggregate_by_category or aggregate_by_neighborhood_type.
    include_geohash_tokens : bool
        If True (or when aggregate_by_context is True), include GH_* tokens in
        the pool of candidates in addition to POI_* tokens.

    Returns
    -------
    pd.DataFrame
        - When no aggregation flags are set: columns
          [token, workiness, poi_category, context, neighborhood_type].
        - When aggregating by category / neighborhood_type: columns
          [token, workiness, n_pois, poi_category, context, neighborhood_type].
        - When aggregate_by_context is True: columns
          [context, workiness, n_tokens, n_pois, n_geohashes].
    """
    wv = model.wv

    # anchor vector: WORK (default) or HOME when target="home"
    anchor = work_anchor
    prefix = work_prefix
    if str(target).lower() == "home" and work_anchor == "WORK" and work_prefix == "WORK_":
        anchor = "HOME"
        prefix = "HOME_"

    if anchor in wv.key_to_index:
        work_vec = wv[anchor]
    else:
        work_tokens = [t for t in wv.key_to_index if t.startswith(prefix)]
        if not work_tokens:
            logger.warning(
                "No anchor tokens found for target '%s' (prefix '%s')", target, prefix
            )
            return pd.DataFrame()
        work_vec = np.mean([wv[t] for t in work_tokens], axis=0)

    # Candidate tokens: POI_* and, optionally, GH_*
    if poi_tokens is None:
        poi_tokens = [t for t in wv.key_to_index if t.startswith("POI_")]
        if aggregate_by_context or include_geohash_tokens:
            poi_tokens += [t for t in wv.key_to_index if t.startswith("GH_")]

    if allowed_day_parts:
        allowed_set = {p.upper() for p in allowed_day_parts}
        poi_tokens = [
            t for t in poi_tokens if _poi_token_allowed_day_part(t, allowed_set)
        ]

    # Compute raw workiness per token
    raw = []
    for tok in poi_tokens:
        if tok in wv.key_to_index:
            sim = 1.0 - cosine_dist(wv[tok], work_vec)
            raw.append({"token": tok, "workiness": sim})

    if not raw:
        return pd.DataFrame()

    poi_meta = _get_poi_meta_lookup()

    # ── Context-level aggregation (POI + GH) ────────────────────────────────
    if aggregate_by_context:
        if aggregate_by_category or aggregate_by_neighborhood_type:
            raise ValueError(
                "aggregate_by_context cannot be combined with "
                "aggregate_by_category or aggregate_by_neighborhood_type"
            )

        gh_context_lookup = _get_gh_context_lookup()

        # Map each token to its context type and group
        ctx_to_items = {}  # context -> list of (workiness, kind)
        for r in raw:
            parsed = _parse_token_for_dims(r["token"])
            ctx = _dim_context(parsed, poi_meta, gh_context_lookup)
            if not ctx:
                continue
            kind = parsed["kind"]
            if ctx not in ctx_to_items:
                ctx_to_items[ctx] = []
            ctx_to_items[ctx].append((r["workiness"], kind))

        rows = []
        for ctx, items in ctx_to_items.items():
            if len(items) < min_n_pois:
                continue
            vals = [x[0] for x in items]
            kinds = [x[1] for x in items]
            n_pois = sum(1 for k in kinds if k == "POI")
            n_geohashes = sum(1 for k in kinds if k == "GH")
            rows.append(
                {
                    "context": ctx,
                    "workiness": np.mean(vals),
                    "n_tokens": len(items),
                    "n_pois": n_pois,
                    "n_geohashes": n_geohashes,
                }
            )
        return pd.DataFrame(rows)

    # ── Token-level output (no aggregation flags) ───────────────────────────
    if not aggregate_by_category and not aggregate_by_neighborhood_type:
        # Add poi_category, context, neighborhood_type from metadata (for POIs)
        for r in raw:
            parsed = _parse_token_for_dims(r["token"])
            meta = poi_meta.get(parsed.get("poi_id") or "") or {}
            r["poi_category"] = meta.get("pois_category") or ""
            r["context"] = meta.get("geohash_context_type") or ""
            r["neighborhood_type"] = meta.get("census_new_neighborhood_type") or ""
            r["keyword"] = meta.get("keyword") or ""
        return pd.DataFrame(raw)

    # ── Aggregation by POI category / neighborhood type ─────────────────────
    neigh_lookup = None  # only POI aggregation here
    agg_to_items = {}  # agg_tok -> list of (workiness, category, context, neighborhood_type)
    for r in raw:
        parsed = _parse_token_for_dims(r["token"])
        meta = poi_meta.get(parsed.get("poi_id") or "") or {}
        cat = meta.get("pois_category") or ""
        ctx = meta.get("geohash_context_type") or ""
        neigh = meta.get("census_new_neighborhood_type") or ""
        agg_token = _build_aggregated_token(
            parsed,
            aggregate_by_neighborhood_type,
            aggregate_by_category,
            False,
            neigh_lookup,
            poi_meta,
            None,
        )
        if agg_token not in agg_to_items:
            agg_to_items[agg_token] = []
        agg_to_items[agg_token].append((r["workiness"], cat, ctx, neigh))

    def _most_common(lst):
        return max(set(lst), key=lst.count) if lst else ""

    rows = []
    for agg_tok, items in agg_to_items.items():
        if len(items) < min_n_pois:
            continue
        vals = [x[0] for x in items]
        cats = [x[1] for x in items if x[1]]
        ctxs = [x[2] for x in items if x[2]]
        neighs = [x[3] for x in items if x[3]]
        rows.append(
            {
                "token": agg_tok,
                "workiness": np.mean(vals),
                "n_pois": len(vals),
                "poi_category": _most_common(cats),
                "context": _most_common(ctxs),
                "neighborhood_type": _most_common(neighs),
            }
        )
    return pd.DataFrame(rows)


def poi_office_geohash_similarity(
    model,
    poi_tokens=None,
    office_context=("office",),
    top_k_per_poi=50,
):
    """
    Compare similarity of POI tokens to all geohash tokens whose context is "office".

    Parameters
    ----------
    model : gensim Word2Vec
    poi_tokens : list[str] or None
        If None, all POI_* tokens in the model are used.
    office_context : tuple[str]
        Context labels (case-insensitive) that count as "office".
    top_k_per_poi : int or None
        If set, keep only top-K office geohashes per POI. If None, return
        all POI–office pairs.

    Returns
    -------
    pd.DataFrame with columns [poi_token, gh_token, similarity].
    """
    wv = model.wv

    # POI candidates
    if poi_tokens is None:
        poi_tokens = [t for t in wv.key_to_index if t.startswith("POI_")]

    if not poi_tokens:
        return pd.DataFrame()

    # Office GH tokens, based on geohash context lookup
    gh_context_lookup = _get_gh_context_lookup()
    office_set = {c.lower() for c in office_context}

    office_tokens = []
    for tok in wv.key_to_index:
        if not tok.startswith("GH_"):
            continue
        parsed = _parse_token_for_dims(tok)
        gh = parsed.get("geohash")
        ctx = gh_context_lookup.get(gh)
        if ctx and ctx.lower() in office_set:
            office_tokens.append(tok)

    if not office_tokens:
        return pd.DataFrame()

    # Use normalised vectors for cosine similarity
    wv_norm = wv.get_normed_vectors()
    poi_idx = [wv.key_to_index[t] for t in poi_tokens]
    gh_idx = [wv.key_to_index[t] for t in office_tokens]

    poi_vecs = wv_norm[poi_idx]
    gh_vecs = wv_norm[gh_idx]

    # Similarity matrix (n_poi x n_office)
    sim = poi_vecs @ gh_vecs.T

    rows = []
    if top_k_per_poi is None:
        for i, p in enumerate(poi_tokens):
            for j, g in enumerate(office_tokens):
                rows.append(
                    {
                        "poi_token": p,
                        "gh_token": office_tokens[j],
                        "similarity": float(sim[i, j]),
                    }
                )
    else:
        k = min(top_k_per_poi, len(office_tokens))
        for i, p in enumerate(poi_tokens):
            row = sim[i]
            # indices of top-k similarities
            idx = np.argpartition(-row, k - 1)[:k]
            idx = idx[np.argsort(-row[idx])]
            for j in idx:
                rows.append(
                    {
                        "poi_token": p,
                        "gh_token": office_tokens[j],
                        "similarity": float(row[j]),
                    }
                )

    return pd.DataFrame(rows)


def compute_workiness(*args, **kwargs):
    """
    Backwards-compatible alias for compute_closeness().
    """
    logger.warning(
        "compute_workiness() is deprecated; use compute_closeness() instead."
    )
    return compute_closeness(*args, **kwargs)


# ── Q3: Neighbourhood Prediction ─────────────────────────────────────────────

def _get_wv(model):
    """Word2Vec has .wv; KeyedVectors is already the vectors."""
    return model.wv if hasattr(model, "wv") else model


# Token names depend on pipeline: GCSCorpus uses HOME/WORK; build_corpus uses TYPE_HOME/TYPE_WORK.
# If the requested token is missing, try these aliases so "home" works with either scheme.
_TOKEN_ALIASES = {"TYPE_HOME": "HOME", "TYPE_WORK": "WORK"}


def _resolve_token(wv, token):
    """Return token if in vocab, else try alias (e.g. TYPE_HOME -> HOME)."""
    if token in wv.key_to_index:
        return token
    alias = _TOKEN_ALIASES.get(token)
    if alias and alias in wv.key_to_index:
        logger.info("Token '%s' not in model; using alias '%s' (model uses HOME/WORK, not TYPE_HOME/TYPE_WORK).", token, alias)
        return alias
    return None


def closest_tokens(model, token, top_k=10, show=True):
    """
    Return the most similar tokens to a given token; optionally print them.

    Parameters
    ----------
    model : Word2Vec or KeyedVectors
        Trained place2vec model.
    token : str
        Token to query. Use "HOME" or "WORK" for GCSCorpus-trained models;
        "TYPE_HOME" / "TYPE_WORK" are auto-mapped to HOME/WORK if missing.
    top_k : int
        Number of nearest neighbours to return (default 10).
    show : bool
        If True, print a ranked list to stdout (default True).

    Returns
    -------
    list of (str, float)
        Pairs (token, similarity) for the top_k closest tokens.
    """
    wv = _get_wv(model)
    resolved = _resolve_token(wv, token)
    if resolved is None:
        logger.warning("Token '%s' not in model. For home/work try 'HOME' or 'WORK' (GCSCorpus models).", token)
        return []
    pairs = wv.most_similar(resolved, topn=top_k)
    if show:
        print(f"Closest {top_k} tokens to '{resolved}':")
        for i, (t, sim) in enumerate(pairs, 1):
            print(f"  {i:2}. {t}  {sim:.4f}")
    return pairs


def get_neighbours(model, anchor_token, top_k=10):
    """
    Return top-K nearest neighbours for an anchor token.

    Returns list of (token, similarity). Accepts Word2Vec or KeyedVectors.
    TYPE_HOME/TYPE_WORK are auto-mapped to HOME/WORK when missing.
    """
    wv = _get_wv(model)
    resolved = _resolve_token(wv, anchor_token)
    if resolved is None:
        logger.warning("Token '%s' not in model. For home/work try 'HOME' or 'WORK'.", anchor_token)
        return []
    return wv.most_similar(resolved, topn=top_k)


def predict_next_tokens(
    model,
    context_tokens,
    top_k=10,
    exclude_context=True,
    show=True,
):
    """
    Predict the next possible token(s) given one or more context tokens.

    Averages the vectors of the context tokens and returns the tokens closest
    to that average (i.e. most likely to appear in the same context / after
    the given sequence).

    Parameters
    ----------
    model : Word2Vec or KeyedVectors
        Trained place2vec model.
    context_tokens : str or list of str
        One token or a sequence. Use "HOME"/"WORK" for GCSCorpus models;
        "TYPE_HOME"/"TYPE_WORK" are auto-mapped to HOME/WORK if missing.
    top_k : int
        Number of predicted tokens to return (default 10).
    exclude_context : bool
        If True, exclude the context tokens from the result (default True).
    show : bool
        If True, print the ranked predictions (default True).

    Returns
    -------
    list of (str, float)
        Pairs (token, similarity) for the top_k predicted tokens.

    Examples
    --------
    >>> place2vec.predict_next_tokens(model, "TYPE_HOME", top_k=5)
    >>> place2vec.predict_next_tokens(model, ["TYPE_HOME", "PART_MORNING"], top_k=5)
    """
    wv = _get_wv(model)
    if isinstance(context_tokens, str):
        context_tokens = [context_tokens]
    context_tokens = [t for t in context_tokens if t]
    if not context_tokens:
        logger.warning("predict_next_tokens: no context tokens provided")
        return []

    # Resolve aliases (e.g. TYPE_HOME -> HOME) so home/work work with either tokenization
    resolved = []
    for t in context_tokens:
        r = _resolve_token(wv, t)
        if r is not None:
            resolved.append(r)
        else:
            logger.warning("Token '%s' not in model (skipped). For home/work try 'HOME' or 'WORK'.", t)
    if not resolved:
        return []
    context_tokens = resolved

    # Average vectors (use normed for cosine similarity)
    indices = [wv.key_to_index[t] for t in context_tokens]
    vectors = wv.get_normed_vectors()[indices]
    avg_vec = np.mean(vectors, axis=0)
    # Renormalize so similar_by_vector gives cosine similarities
    norm = np.linalg.norm(avg_vec)
    if norm > 1e-9:
        avg_vec = avg_vec / norm

    # Ask for extra candidates if we will filter out context
    n_request = top_k + len(context_tokens) if exclude_context else top_k
    context_set = set(context_tokens)
    pairs = wv.similar_by_vector(avg_vec, topn=n_request)

    if exclude_context:
        pairs = [(t, s) for t, s in pairs if t not in context_set]
    pairs = pairs[:top_k]

    if show:
        ctx_str = " + ".join(context_tokens) if len(context_tokens) <= 3 else f"{len(context_tokens)} tokens"
        print(f"Next tokens predicted from context [{ctx_str}] (top {top_k}):")
        for i, (t, sim) in enumerate(pairs, 1):
            print(f"  {i:2}. {t}  {sim:.4f}")
    return pairs


def save_location_similarities(
    model,
    table_name="place2vec_location_similarities",
    dataset="phd_datasets",
    project=PROJECT,
    period_name=None,
    top_k=100,
    token_prefix=None,
    if_table_exist="replace",
):
    """
    For each location (token) in the model, compute top-K most similar locations
    and save to BigQuery.

    Parameters
    ----------
    model : gensim Word2Vec
    table_name : str
        BigQuery table name. Default "place2vec_location_similarities".
    dataset : str
        BigQuery dataset. Default "phd_datasets".
    project : str
    period_name : str or None
        If provided, adds a 'period' column (e.g. "pre_covid").
    top_k : int
        Number of closest neighbours per location. Default 100.
    token_prefix : str or None
        If provided, only process tokens with this prefix (e.g. "POI_", "GH_").
        None = all tokens.
    if_table_exist : str
        "replace" (default) or "append". Use "append" when saving multiple periods
        to the same table.

    Output table columns: anchor_token, neighbour_token, rank, similarity_score, period
    """
    wv = model.wv
    keys = list(wv.index_to_key)
    if token_prefix:
        keys = [k for k in keys if k.startswith(token_prefix)]
    if not keys:
        logger.warning("No tokens found (prefix=%s)", token_prefix)
        return

    logger.info("Computing top-%d similarities for %d locations ...", top_k, len(keys))
    indices = [wv.key_to_index[k] for k in keys]
    vectors = wv.get_normed_vectors()[indices]
    sim_matrix = np.dot(vectors, vectors.T)

    n = len(keys)
    k_total = min(top_k + 1, n)
    partitioned = np.argpartition(-sim_matrix, k_total - 1, axis=1)[:, :k_total]
    row_idx = np.arange(n)[:, None]
    partitioned_scores = sim_matrix[row_idx, partitioned]
    sorted_idx = np.argsort(-partitioned_scores, axis=1)
    top_indices = np.take_along_axis(partitioned, sorted_idx, axis=1)
    top_scores = np.take_along_axis(partitioned_scores, sorted_idx, axis=1)

    anchor = np.repeat(keys, k_total)
    neighbour = np.array(keys)[top_indices.ravel()]
    scores = top_scores.ravel().astype(float)

    df = pd.DataFrame({
        "anchor_token": anchor,
        "neighbour_token": neighbour,
        "similarity_score": scores,
    })
    df = df[df["anchor_token"] != df["neighbour_token"]].copy()
    df["rank"] = df.groupby("anchor_token").cumcount() + 1
    df = df[df["rank"] <= top_k]

    if period_name:
        df["period"] = period_name

    utils.upload_table_to_bq(df, dataset, table_name, project=project, if_table_exist=if_table_exist)
    logger.info("Saved %d similarity rows → %s.%s.%s", len(df), project, dataset, table_name)
    return df


# ── Q4: 15-Minute City Test (Geo vs Functional Distance) ─────────────────────

def correlation_geo_func(
    model,
    pairs,
    geohash_coords,
):
    """
    For a set of (HOME_token, POI_token) pairs compute:
      - geo_dist  : haversine km  (looked up via geohash_coords)
      - func_dist : 1 - cosine_sim

    Parameters
    ----------
    model : gensim Word2Vec
    pairs : list of (home_token, poi_token)
    geohash_coords : dict  {token: (lat, lon)}

    Returns
    -------
    pd.DataFrame  with [home, poi, geo_dist_km, func_dist, cosine_sim]
    """
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    wv = model.wv
    rows = []
    for home_tok, poi_tok in pairs:
        if home_tok not in wv.key_to_index or poi_tok not in wv.key_to_index:
            continue
        if home_tok not in geohash_coords or poi_tok not in geohash_coords:
            continue

        sim = 1.0 - cosine_dist(wv[home_tok], wv[poi_tok])
        func_d = 1.0 - sim

        h_lat, h_lon = geohash_coords[home_tok]
        p_lat, p_lon = geohash_coords[poi_tok]
        geo_d = haversine(h_lat, h_lon, p_lat, p_lon)

        rows.append({
            "home": home_tok,
            "poi": poi_tok,
            "geo_dist_km": geo_d,
            "func_dist": func_d,
            "cosine_sim": sim,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        corr = df["geo_dist_km"].corr(df["func_dist"])
        logger.info("Geo-Func Pearson correlation = %.4f  (N=%d)", corr, len(df))
    return df
