-- Google POIs remote-work flags pipeline (no UPDATE statements)
-- Output: phd-habidatum.phd_pois.google_pois_flags
-- Period for behavior flag: 2021-01-01 to 2023-12-31
-- Each step uses: SELECT * + one newly computed flag column.

-- ============================================================================
-- 0) Base table (Google only, initialize flag columns to NULL)
-- ============================================================================
CREATE OR REPLACE TABLE `phd-habidatum.phd_pois.google_pois_flags`
CLUSTER BY geometry AS
WITH google_base AS (
  SELECT
    CAST(g.id AS STRING) AS id,
    g.displayname.text AS name,
    LOWER(g.primarytype) AS primarytype,
    g.geometry,
    gs.geohash,
    gsa.yishuv_stat_2022,
    gsa.context_type AS geohash_context_type,
    gsa.main_function_txt AS census_main_function_txt,
    sa.new_neighborhood_type AS census_new_neighborhood_type
  FROM `phd-habidatum.phd_pois.google_3rdplaces_cities` g
  JOIN `phd-habidatum.phd_adm.geohashes_with_sa_gushdan` gs
    ON ST_WITHIN(g.geometry, gs.geometry)
  LEFT JOIN `phd-habidatum.phd_adm.geohashes_with_sa_gushdan_context` gsa
    USING (geohash)
  LEFT JOIN `phd-habidatum.phd_datasets.sa_neigborhoods_types_new` sa
    ON CAST(sa.yishuv_stat_2022 AS STRING) = CAST(gsa.yishuv_stat_2022 AS STRING)
)
SELECT
  b.*,
  CAST(NULL AS BOOL) AS flag_category_suits_rw,
  CAST(NULL AS BOOL) AS flag_location_suits_rw,
  CAST(NULL AS BOOL) AS flag_review_evidence,
  CAST(NULL AS BOOL) AS flag_atmosphere_suits_rw,
  CAST(NULL AS BOOL) AS flag_user_behavior_suits_rw,
  CAST(NULL AS BOOL) AS flag_groundtruth,
  CAST(NULL AS TIMESTAMP) AS flag_category_updated_at,
  CAST(NULL AS TIMESTAMP) AS flag_location_updated_at,
  CAST(NULL AS TIMESTAMP) AS flag_review_updated_at,
  CAST(NULL AS TIMESTAMP) AS flag_atmosphere_updated_at,
  CAST(NULL AS TIMESTAMP) AS flag_user_behavior_updated_at,
  CAST(NULL AS TIMESTAMP) AS flag_groundtruth_updated_at,
  CURRENT_TIMESTAMP() AS created_at,
  CURRENT_TIMESTAMP() AS updated_at
FROM google_base b;

-- ============================================================================
-- 1) Flag: POI category suits remote work
--    (SELECT * + new feature, no UPDATE)
-- ============================================================================
CREATE OR REPLACE TABLE `phd-habidatum.phd_pois.google_pois_flags`
CLUSTER BY geometry AS
SELECT
  * EXCEPT(flag_category_suits_rw, flag_category_updated_at, updated_at),
  CASE
    WHEN primarytype IS NULL THEN NULL
    WHEN primarytype IN (
      'cafe', 'coffee_shop', 'tea_house', 'bakery', 'book_store', 'library',
      'community_center', 'cultural_center', 'museum', 'dessert_restaurant',
      'breakfast_restaurant', 'brunch_restaurant', 'cat_cafe', 'dog_cafe'
    ) THEN TRUE
    WHEN primarytype IN (
      'office', 'corporate_office', 'coworking_space', 'university',
      'college', 'school', 'hospital', 'medical_center', 'train_station',
      'bus_station', 'airport', 'government_office'
    ) THEN FALSE
    ELSE NULL
  END AS flag_category_suits_rw,
  CURRENT_TIMESTAMP() AS flag_category_updated_at,
  CURRENT_TIMESTAMP() AS updated_at
FROM `phd-habidatum.phd_pois.google_pois_flags`;

-- ============================================================================
-- 2) Flag: Location suits remote work (not office/university)
--    (SELECT * + new feature, no UPDATE)
-- ============================================================================
CREATE OR REPLACE TABLE `phd-habidatum.phd_pois.google_pois_flags`
CLUSTER BY geometry AS
WITH location_calc AS (
  SELECT
    p.id,
    CASE
      WHEN p.geohash IS NULL THEN NULL
      WHEN gf.geohash IS NULL THEN NULL
      WHEN (
        COALESCE(gf.count_universities, 0)
        + COALESCE(gf.count_colleges, 0)
        + COALESCE(gf.count_schools, 0)
        + COALESCE(gf.count_medical_institutions, 0)
        + COALESCE(gf.count_transport_hubs, 0)
      ) = 0
      AND COALESCE(gf.flag_work_zone, 0) = 0 THEN TRUE
      ELSE FALSE
    END AS flag_location_suits_rw
  FROM `phd-habidatum.phd_pois.google_pois_flags` p
  LEFT JOIN `phd-habidatum.phd_datasets.geohash_features` gf
    USING (geohash)
)
SELECT
  p.* EXCEPT(flag_location_suits_rw, flag_location_updated_at, updated_at),
  l.flag_location_suits_rw AS flag_location_suits_rw,
  CURRENT_TIMESTAMP() AS flag_location_updated_at,
  CURRENT_TIMESTAMP() AS updated_at
FROM `phd-habidatum.phd_pois.google_pois_flags` p
LEFT JOIN location_calc l
  USING (id);

-- ============================================================================
-- 3) Flag: Review evidence (direct RW keyword evidence)
--    (SELECT * + new feature, no UPDATE)
-- ============================================================================
CREATE OR REPLACE TABLE `phd-habidatum.phd_pois.google_pois_flags`
CLUSTER BY geometry AS
WITH review_calc AS (
  SELECT
    p.id,
    CASE
      WHEN r.id IS NULL THEN NULL
      ELSE CAST(r.flag_direct_keyword AS BOOL)
    END AS flag_review_evidence
  FROM `phd-habidatum.phd_pois.google_pois_flags` p
  LEFT JOIN `phd-habidatum.phd_pois.google_3rd_places_rw_review` r
    ON p.id = CAST(r.id AS STRING)
)
SELECT
  p.* EXCEPT(flag_review_evidence, flag_review_updated_at, updated_at),
  r.flag_review_evidence AS flag_review_evidence,
  CURRENT_TIMESTAMP() AS flag_review_updated_at,
  CURRENT_TIMESTAMP() AS updated_at
FROM `phd-habidatum.phd_pois.google_pois_flags` p
LEFT JOIN review_calc r
  USING (id);

-- ============================================================================
-- 4) Flag: Atmosphere suits remote work
--    (SELECT * + new feature, no UPDATE)
-- ============================================================================
CREATE OR REPLACE TABLE `phd-habidatum.phd_pois.google_pois_flags`
CLUSTER BY geometry AS
WITH atmosphere_calc AS (
  SELECT
    p.id,
    CASE
      WHEN r.id IS NULL THEN NULL
      ELSE CAST(r.flag_atmosphere AS BOOL)
    END AS flag_atmosphere_suits_rw
  FROM `phd-habidatum.phd_pois.google_pois_flags` p
  LEFT JOIN `phd-habidatum.phd_pois.google_3rd_places_rw_review` r
    ON p.id = CAST(r.id AS STRING)
)
SELECT
  p.* EXCEPT(flag_atmosphere_suits_rw, flag_atmosphere_updated_at, updated_at),
  a.flag_atmosphere_suits_rw AS flag_atmosphere_suits_rw,
  CURRENT_TIMESTAMP() AS flag_atmosphere_updated_at,
  CURRENT_TIMESTAMP() AS updated_at
FROM `phd-habidatum.phd_pois.google_pois_flags` p
LEFT JOIN atmosphere_calc a
  USING (id);

-- ============================================================================
-- 5) Flag: Users behavior suits remote work
--    Reuses rw_share logic from 1100-third-places-users-signals:
--    rw_share = n_long_morning_weekday_visits / n_visits
--    (SELECT * + new feature, no UPDATE)
-- ============================================================================
CREATE OR REPLACE TABLE `phd-habidatum.phd_pois.google_pois_flags`
CLUSTER BY geometry AS
WITH behavior_2021_2023 AS (
  SELECT
    CAST(poi_id AS STRING) AS id,
    SUM(n_visits) AS n_visits_total,
    SUM(n_visits_weekday) AS n_visits_weekday_total,
    SUM(n_long_morning_weekday_visits) AS n_long_morning_weekday_visits_total,
    SAFE_DIVIDE(SUM(n_long_morning_weekday_visits), SUM(n_visits)) AS rw_share
  FROM `phd-habidatum.phd_datasets.pois_visits_monthly_19022026`
  WHERE worker_type = 'remote_workers'
    AND month >= DATE('2021-01-01')
    AND month <= DATE('2023-12-31')
  GROUP BY CAST(poi_id AS STRING)
),
behavior_calc AS (
  SELECT
    p.id,
    CASE
      WHEN b.id IS NULL THEN NULL
      WHEN b.n_visits_weekday_total = 0 THEN FALSE
      WHEN b.rw_share >= 0.20 THEN TRUE
      ELSE FALSE
    END AS flag_user_behavior_suits_rw
  FROM `phd-habidatum.phd_pois.google_pois_flags` p
  LEFT JOIN behavior_2021_2023 b
    USING (id)
)
SELECT
  p.* EXCEPT(flag_user_behavior_suits_rw, flag_user_behavior_updated_at, updated_at),
  b.flag_user_behavior_suits_rw AS flag_user_behavior_suits_rw,
  CURRENT_TIMESTAMP() AS flag_user_behavior_updated_at,
  CURRENT_TIMESTAMP() AS updated_at
FROM `phd-habidatum.phd_pois.google_pois_flags` p
LEFT JOIN behavior_calc b
  USING (id);

-- ============================================================================
-- 6) Flag: Groundtruth indication
--    Reuses matching logic from 102-create-full-rw-places-dataset:
--    ST_DWITHIN 40m + best name edit distance per groundtruth place
--    (SELECT * + new feature, no UPDATE)
-- ============================================================================
CREATE OR REPLACE TABLE `phd-habidatum.phd_pois.google_pois_flags`
CLUSTER BY geometry AS
WITH gt_matches AS (
  SELECT
    CAST(g.id AS STRING) AS id,
    ROW_NUMBER() OVER (
      PARTITION BY t.name
      ORDER BY EDIT_DISTANCE(LOWER(t.name), LOWER(g.displayname.text))
    ) AS rnk
  FROM `phd-habidatum.phd_pois.third_rw_places_groundtruth` t
  JOIN `phd-habidatum.phd_pois.google_3rdplaces_cities` g
    ON ST_DWITHIN(t.geometry, g.geometry, 40)
),
gt_ids AS (
  SELECT DISTINCT id
  FROM gt_matches
  WHERE rnk = 1
),
groundtruth_calc AS (
  SELECT
    p.id,
    CASE
      WHEN p.geometry IS NULL THEN NULL
      WHEN gt.id IS NOT NULL THEN TRUE
      ELSE FALSE
    END AS flag_groundtruth
  FROM `phd-habidatum.phd_pois.google_pois_flags` p
  LEFT JOIN gt_ids gt
    USING (id)
)
SELECT
  p.* EXCEPT(flag_groundtruth, flag_groundtruth_updated_at, updated_at),
  g.flag_groundtruth AS flag_groundtruth,
  CURRENT_TIMESTAMP() AS flag_groundtruth_updated_at,
  CURRENT_TIMESTAMP() AS updated_at
FROM `phd-habidatum.phd_pois.google_pois_flags` p
LEFT JOIN groundtruth_calc g
  USING (id);

