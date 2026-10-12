# Metrics
from typing import Literal, Optional
import pandas as pd
from geofunctions import utils
from loguru import logger
import numpy as np


class BuildingsPopulation:
    """
    Calculates building-level metrics, including the number of households, points of interest (POIs),
    and enterprises located in each building.

    Input data should include 3 layers:
     1. building footprint with height
     2. layer with building class: commercial/residential
     3. layer with population number by block group

    Attributes
    ----------
    buildings_table : str
        Name of the table containing building details (e.g., from Maxar data).

    classification_table : str
        Name of the table with building classifications (e.g., from Google Maps).
        Must include the 'is_commercial' column.

    pop_table : str
        Name of the population table.

    borders_table : str, optional
        Name of the table containing border geometries.

    output_table : str, optional
        Name of the output table for storing results.

    pop_column : str
        Name of the column containing population data.

    hh_column : str
        Name of the column containing household data.

    id_build_column : str
        Name of the building ID column.

    id_block_column : str
        Name of the block ID column.

    the_geom_column : str, default="the_geom"
        Name of the geometry column in the buildings table.

    height_column : str, default="AGL"
        Name of the building height column.

    floor_height : float, default=3
        Estimated height of each floor in meters.

    basement_height : float, default=4
        Estimated height of the basement in meters.

    max_private_house_height : float, default=6
        Maximum height (in meters) for a building to be considered a private house.

    if_table_exists : {"fail", "skip", "replace", "append"}, default="fail"
        Behavior if the output table already exists.


    stats : bool, default=False
        If True, collects additional statistics during execution.

    desc : str, optional
        Description of the processing step.

    base_path : str, optional
        Base path for storing outputs.

    Methods
    -------
    get_builds_type():
        Fetches building types from classification data.

    calculate_floor_area():
        Calculates the living floor area for each building.

    check_if_private():
        Determines whether a building is classified as private.

    get_resid_builds():
        Retrieves details of residential buildings.

    get_builds_metrics():
        Joins data from multiple sources to produce complete building metrics.

    summarize_buildings_stats(df):
        Returns summary statistics for the building dataset.

    run():
        Executes all processing steps and outputs final metrics.

    Notes
    -----
    All required tables must be accessible to the client before running.
    """

    def __init__(
        self,
        buildings_table: str,
        pop_table: str,
        pop_column: str = "pop",
        hh_column: str = "hh",
        id_build_column: str = "build_id",
        id_block_column: str = "block_id",
        flag_commercial_column: str = "is_commercial",
        the_geom_column: str = "the_geom",
        height_column: str = "AGL",
        floor_height: float = 3,
        basement_height: float = 4,
        max_private_house_height: float = 6,
        max_area_per_hh_tall: float = 130.0,  # min m² per HH for tall buildings
        max_area_per_hh_low: float = 250.0,  # min m² per HH for low buildings
        min_area_per_hh: float = 30,  # max households per 1 m² living floor
        output_table: Optional[str] = None,
        borders_table: Optional[str] = None,
        classification_table: Optional[str] = None,
        stats: bool = False,
        if_table_exists: Literal["fail", "skip", "replace", "append"] = "fail",
        desc: Optional[str] = None,
        base_path: Optional[str] = None,
    ):

        self.buildings_table = buildings_table
        self.classification_table = classification_table
        self.pop_table = pop_table
        self.borders_table = borders_table
        self.output_table = output_table
        self.pop_column = pop_column
        self.hh_column = hh_column
        self.flag_commercial_column = flag_commercial_column
        self.floor_height = floor_height
        self.basement_height = basement_height
        self.max_private_house_height = max_private_house_height
        self.max_area_per_hh_tall = max_area_per_hh_tall
        self.max_area_per_hh_low = max_area_per_hh_low
        self.min_area_per_hh = min_area_per_hh
        self.the_geom_column = the_geom_column
        self.id_build_column = id_build_column
        self.id_block_column = id_block_column
        self.height_column = height_column
        self.if_table_exists = if_table_exists

    def get_builds_type(self):
        """
        Returns dataset build_id, the_geom, is_commercial based on google maps data
        """
        if self.classification_table is not None:
            query_google_build_type = f"""
        -- commercial/residential building
                WITH
                ms_builds AS(
                    SELECT

                    f.{self.the_geom_column},
                    f.{self.id_build_column},
                    FROM {self.buildings_table} f
                    {self._sql_borders_join()}
                    ),

                    inter_type AS (
                    SELECT  b.{self.id_build_column},
                            {self.flag_commercial_column} is_commercial,
                            st_area(st_intersection(b.the_geom, r.the_geom))/st_area(b.the_geom) as share_inter_area,
                            b.{self.the_geom_column}
                    FROM ms_builds b,
                    {self.classification_table} r
                    WHERE ST_INTERSECTS(b.{self.the_geom_column}, r.the_geom)
                    ),

                    rank_type as (
                    SELECT  r.*,
                        inter_type.* except({self.id_build_column}, {self.the_geom_column}),
                        RANK() OVER (PARTITION BY {self.id_build_column} ORDER BY share_inter_area desc) rank
                    FROM ms_builds r
                    LEFT JOIN  inter_type   USING({self.id_build_column})
                ),
                buildings_type as (
                    SELECT * except(rank)  FROM rank_type
                    WHERE rank=1
                )
                SELECT
                *
                EXCEPT(share_inter_area, is_commercial),
                COALESCE (is_commercial, False) as is_commercial # if not recognised from ggmaps then residential
                FROM  buildings_type
        """
        else:
            query_google_build_type = f"""select f.{self.the_geom_column},
                    f.{self.id_build_column},
                     COALESCE ({self.flag_commercial_column}, False) as is_commercial,
                    FROM {self.buildings_table} f
                    {self._sql_borders_join()}"""
        logger.info(query_google_build_type)
        df_type = utils.read_query_to_dataframe(query_google_build_type)
        logger.info(f"downloaded the table of size {df_type.shape}")
        df_type = df_type.drop_duplicates()
        return df_type

    def calculate_floor_area(self, height: float, area: float):
        """Returns living floor area per each building."""
        if area < 40:
            return 1, 0  # not inhabited building
        else:
            num_floors = (height - self.basement_height) // self.floor_height
            if num_floors >= 16:
                num_floors = (height - self.basement_height) // (self.floor_height + 1)
            flg_private = self.check_if_private(height)
            if flg_private:
                return 1, area
        return num_floors, area * num_floors

    def check_if_private(self, height: float):
        """Checks if a building is private based on height."""
        if height <= self.max_private_house_height:
            return 1
        return 0

    # ---------- helpers ----------
    def _sql_borders_join(self) -> str:
        if not self.borders_table:
            return ""
        return f"JOIN {self.borders_table} l ON ST_WITHIN(ST_CENTROID(f.{self.the_geom_column}), l.the_geom)"

    def _query_buildings_with_pop(self) -> pd.DataFrame:
        block_id_col = self.id_block_column
        sql = f"""
            SELECT
                b.{self.id_build_column} AS {self.id_build_column},
                ST_AREA(b.{self.the_geom_column}) AS area,
                CAST(b.{self.height_column} AS FLOAT64) AS height,
                f.{block_id_col} AS block_id
            FROM {self.buildings_table} b
            JOIN {self.pop_table} f
            ON ST_WITHIN(ST_CENTROID(b.{self.the_geom_column}), f.{self.the_geom_column})
            {self._sql_borders_join()}
        """
        logger.info("start uploading population data")
        return utils.read_query_to_dataframe(sql)

    def _attach_types(
        self, df_builds_big: pd.DataFrame, df_type: pd.DataFrame
    ) -> pd.DataFrame:
        return df_builds_big.merge(
            df_type[[self.id_build_column, "is_commercial"]],
            on=self.id_build_column,
            how="left",
        )

    def _calc_lfa_for_residential(self, df: pd.DataFrame) -> pd.DataFrame:
        df["living_floor_area"] = 0.0
        df["num_floors"] = 0
        mask = ~df["is_commercial"].fillna(False)
        if mask.any():
            t = df.loc[mask, ["height", "area"]].apply(
                lambda r: self.calculate_floor_area(
                    float(r["height"]), float(r["area"])
                ),
                axis=1,
                result_type="expand",
            )
            df.loc[mask, ["num_floors", "living_floor_area"]] = t.values
        return df

    def _block_totals(self) -> pd.DataFrame:
        block_id_col = self.id_block_column
        sql = f"""
            SELECT
                {block_id_col} AS block_id,
                SUM({self.pop_column}) AS total_exp_pop,
                SUM({self.hh_column})  AS total_num_households,
                SUM({self.pop_column})/NULLIF(SUM({self.hh_column}),0) AS household_size
            FROM {self.pop_table}
            GROUP BY {block_id_col}
        """
        exp = utils.read_query_to_dataframe(sql).set_index("block_id")
        # guard tiny/invalid sizes
        exp["household_size"] = exp["household_size"].apply(
            lambda x: 1.2 if (pd.notna(x) and 0 < x < 1) else x
        )
        return exp

    def _apply_density_bounds(
        self,
        df: pd.DataFrame,
        height_thresh: float = 12.0,
    ) -> pd.DataFrame:
        # tall
        is_tall = df["height"] > height_thresh
        if is_tall.any():
            lb_tall = np.maximum(
                1.0 / df.loc[is_tall, "area"], 1.0 / self.max_area_per_hh_tall
            )
            df.loc[is_tall, "num_hh_1m_lfr"] = df.loc[is_tall, "num_hh_1m_lfr"].clip(
                lower=lb_tall, upper=1.0 / self.min_area_per_hh
            )

        # small/low
        is_small = ~is_tall
        if is_small.any():
            lb_small = np.maximum(
                1.0 / df.loc[is_small, "area"], 1.0 / self.max_area_per_hh_low
            )
            df.loc[is_small, "num_hh_1m_lfr"] = df.loc[is_small, "num_hh_1m_lfr"].clip(
                lower=lb_small, upper=1.0 / self.min_area_per_hh
            )

        return df

    def _finalize_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        # households per building
        raw = df["num_hh_1m_lfr"] * df["living_floor_area"]
        df["count_hh"] = raw.apply(
            lambda x: 1 if 0 < x < 1 else int(round(x))
        )  # ensure at least 1 if >0
        # population per building
        df["pop_building"] = df["count_hh"] * df["household_size"]
        # m² per HH
        df["lfa_per_hh"] = np.where(
            df["count_hh"] > 0, df["living_floor_area"] / df["count_hh"], np.nan
        )
        # clean
        return df.drop(columns=["is_commercial"], errors="ignore")

    def get_resid_builds(self, df_type: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate households in residential buildings using building size and block-level population,
        returning per-building metrics.
        """
        # 1) buildings + block id
        df_builds_big = self._query_buildings_with_pop()
        logger.info(f"buildings pop data was uploaded. Size:{df_builds_big.shape}")
        # 2) attach types & compute LFA for residential
        df_builds_big = self._attach_types(df_builds_big, df_type)
        df_builds_big["height"] = df_builds_big["height"].astype(float)
        df_builds_big = self._calc_lfa_for_residential(df_builds_big)

        # 3) block-level totals & join
        block_lfa = (
            df_builds_big.groupby("block_id", dropna=False)["living_floor_area"]
            .sum()
            .rename("block_living_floor_area")
        )
        exp_totals = self._block_totals()
        exp_totals = exp_totals.join(block_lfa, how="left")

        df_builds_big = df_builds_big.set_index("block_id").join(exp_totals, how="left")

        # 4) households density per m² living floor
        df_builds_big["num_hh_1m_lfr"] = np.where(
            (df_builds_big["block_living_floor_area"] > 0)
            & pd.notna(df_builds_big["total_num_households"]),
            df_builds_big["total_num_households"]
            / df_builds_big["block_living_floor_area"],
            0.0,
        )

        # 5) clamp densities by building type
        df_builds_big = self._apply_density_bounds(df_builds_big)

        # 6) finalize per-building counts & tidy
        df_builds_big = df_builds_big.reset_index()
        df_builds_big = self._finalize_counts(df_builds_big)

        return df_builds_big

    def get_buildings_dataframe_output(self):
        """
        Joins building type and residential building dataframes on the building ID column.

        This method retrieves two dataframes: one containing building types and another containing residential building information.
        It performs a left join on the building ID column to combine these dataframes, filling any resulting missing values with zero.
        Returns the merged dataframe.

        Returns:
            pandas.DataFrame: The merged dataframe containing building type and residential building information.
        """
        """Join 2 tables together"""
        df_type = self.get_builds_type()
        df_hh = self.get_resid_builds(df_type)

        df_build = df_type.merge(df_hh, how="left", on=self.id_build_column)

        df_build = df_build.fillna(0)

        return df_build

    def summarize_buildings_stats(self, df: pd.DataFrame) -> pd.Series:
        """
        Summary statistics for buildings:
        - Number of private houses (based on self.max_private_house_height)
        - Number of buildings up to 5 floors
        - Share of residential / commercial buildings
        - Average living floor area (LFA) per household (global)
        - Number of buildings with LFA/HH < 40
        - Number of buildings with LFA/HH > 150

        Expected columns in df:
        ['height', 'num_floors', 'is_commercial', 'living_floor_area', 'count_hh', 'lfa_per_hh'].

        All calculations skip NaN values where appropriate.
        """
        if df.empty:
            return pd.Series(
                {
                    "n_buildings": 0,
                    "n_private": 0,
                    "n_up_to_5_floors": 0,
                    "share_residential": 0.0,
                    "share_commercial": 0.0,
                    "avg_lfa_per_hh": float("nan"),
                    "n_lfa_per_hh_lowthan_40": 0,
                    "n_lfa_per_hh_greaterthan_150": 0,
                }
            )

        # Private houses by height threshold (default 6.0 m if not set)
        max_priv = self.max_private_house_height
        m_private = (
            pd.to_numeric(df.get("height"), errors="coerce").fillna(-1) <= max_priv
        )

        # Buildings up to 5 floors
        m_upto5 = pd.to_numeric(df.get("num_floors"), errors="coerce").fillna(-1) <= 5

        # Residential / Commercial shares
        is_comm = df.get("is_commercial")
        if is_comm is None:
            is_comm = pd.Series(False, index=df.index)
        is_comm = is_comm.fillna(False).astype(bool)
        n_total = len(df)
        n_comm = int(is_comm.sum())
        n_res = int((~is_comm).sum())
        share_comm = n_comm / n_total if n_total else 0.0
        share_res = n_res / n_total if n_total else 0.0

        # Average LFA per HH (global, weighted by households)
        lfa = pd.to_numeric(df.get("living_floor_area"), errors="coerce")
        hh = pd.to_numeric(df.get("count_hh"), errors="coerce")
        lfa_sum = lfa.where(lfa.notna(), 0).sum()
        hh_sum = hh.where((hh.notna()) & (hh > 0), 0).sum()
        avg_lfa_per_hh_global = (lfa_sum / hh_sum) if hh_sum > 0 else float("nan")

        # Threshold counts for LFA/HH
        lfa_per_hh = pd.to_numeric(df.get("lfa_per_hh"), errors="coerce")
        n_lt_40 = int((lfa_per_hh < 40).sum(skipna=True))
        n_gt_150 = int((lfa_per_hh > 150).sum(skipna=True))

        return pd.Series(
            {
                "Num buildings": int(n_total),
                "Num private buildings": int(m_private.sum()),
                "Num up to 5 floors buildings": int(m_upto5.sum()),
                "Share residential buildings": float(share_res),
                "Share commercial buildings": float(share_comm),
                "AVG living area per HH": float(avg_lfa_per_hh_global),
                "Num HH with living area low than 40": n_lt_40,
                "Num HH with living area greater than 150": n_gt_150,
            }
        )

    def store_tables(self):
        """ """
        df2db = self.buildings_features_table.reset_index()
        df2db["build_type"] = df2db["is_commercial"].map(
            {True: "Commercial", False: "Residential"}
        )
        utils.upload_table_to_bq(
            df2db, self.output_table, if_table_exists=self.if_table_exists
        )

    def get_summary_output(self):
        return self.summarize_buildings_stats(self.buildings_features_table)

    def run(self):
        logger.info("start uploading data")
        self.buildings_features_table = self.get_buildings_dataframe_output()
        results = self.get_summary_output()
        logger.info("============")
        logger.info("Building statistics")
        logger.info(results)
        if self.output_table is not None:
            logger.info("start uploading table to BQ")
            self.store_tables()
        return self.buildings_features_table
