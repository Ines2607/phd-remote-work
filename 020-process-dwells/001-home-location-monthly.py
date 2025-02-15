"""
This script processes dwelling data to identify home locations and analyze mobility patterns.
It performs the following main tasks:
1. Filters and processes dwelling data for a specified month
2. Identifies home locations based on nighttime and Shabbat activity patterns
3. Calculates various metrics including:
   - Spatial variance of movements
   - Work-from-home patterns
   - Population correlations
   - User frequency statistics
4. Saves results to geojson and csv files with summary statistics
"""

import os
import argparse
import numpy as np
import geopandas as gpd
import pandas as pd
from geofunctions import utils


def calculate_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency statistics for home and work hours per user.

    Args:
        df (pd.DataFrame): DataFrame containing hourly dwelling data with the following columns:
            - identifier: unique user ID
            - hour: hour of the day (0-23)
            - flag_home_hour: boolean indicating if hour is typically spent at home
            - flag_work_hour: boolean indicating if hour is typically spent at work

    Returns:
        pd.DataFrame: DataFrame containing frequency counts per user with columns:
            - identifier: unique user ID
            - freq_home_hours_count: number of frequent home hours
            - freq_work_hours_count: number of frequent work hours
            
    Notes:
        A "frequent" hour is defined as occurring at least 2 times within the month
        for a given user at a given hour of day.
    """
    # Compute frequencies for home hours
    home_hours_freq = (
        df.loc[df["flag_home_hour"]]
        .groupby(["identifier", "hour"])
        .size()
        .reset_index(name="count")
    )
    # Keep only those home hours for each user that occured at least 2 times within a month
    freq_home_hours_count = (
        home_hours_freq.loc[home_hours_freq["count"] >= 2]
        .groupby("identifier")
        .size()
        .reset_index(name="freq_home_hours_count")
    )

    # Compute frequencies for work hours
    work_hours_freq = (
        df.loc[df["flag_work_hour"]]
        .groupby(["identifier", "hour"])
        .size()
        .reset_index(name="count")
    )

    # Keep only those work hours for each user that occured at least 2 times within a month
    freq_work_hours_count = (
        work_hours_freq.loc[work_hours_freq["count"] >= 2]
        .groupby("identifier")
        .size()
        .reset_index(name="freq_work_hours_count")
    )

    # Merge home and work frequencies
    result = pd.merge(
        freq_home_hours_count, 
        freq_work_hours_count, 
        on="identifier", 
        how="inner"
    ).fillna(0)

    return result

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Read a Parquet file with a specified month filter."
)
parser.add_argument(
    "--month", type=str, default=1, help="Specify the number of files to process"
)
args = parser.parse_args()
MONTH = args.month

# Set up variables
path_dir = utils.get_path("raw", "dwells")
dict_key_metrics = dict()


df_filtered_good = pd.read_parquet(f"{path_dir}/data_filtered_{MONTH}.parquet")
if df_filtered_good.empty:
    raise ValueError(f"No data found for month {MONTH}")

df_filtered_dwells = utils.make_gdf(df_filtered_good, geometry="the_geom")
print(f"Read file successfully. Shape: {df_filtered_dwells.shape}")

# user-geohash statistic
user_geohash_desc = (
    df_filtered_dwells.groupby(["identifier", "geohash"])["duration_hours"]
    .sum()
    .to_frame()
    .join(
        df_filtered_dwells.groupby(["identifier", "geohash"])["date"]
        .nunique()
        .to_frame()
    )
)

user_geohash_desc["rank_frequency_day"] = user_geohash_desc.groupby(["identifier"])[
    "date"
].rank(ascending=False, method="first")

# keep only top 5 locations for each user
df_filtered_dwells = df_filtered_dwells.merge(
    user_geohash_desc[user_geohash_desc["rank_frequency_day"] <= 5].reset_index()[
        ["identifier", "geohash"]
    ],
    how="inner",
    on=["identifier", "geohash"],
)
if df_filtered_dwells.empty:
    raise ValueError("No data remaining after filtering top 5 geohashes")
print(f"Top 5 geohashes left for user. Shape: {df_filtered_dwells.shape}")

df_filtered_dwells["date"] = pd.to_datetime(df_filtered_dwells["date"])

# explode dwells to hourly signals
df_filtered_dwells["hours_in_interval"] = df_filtered_dwells.apply(
    lambda row: pd.date_range(
        start=row["start_date_time"].floor("h"),
        end=row["end_date_time"].ceil("h") - pd.Timedelta(hours=1),
        freq="h",
    ).tolist(),
    axis=1,
)

# Explode the hours into separate rows
df_hours = df_filtered_dwells.explode("hours_in_interval")
df_hours["hour"] = df_hours["hours_in_interval"].dt.hour

print("exploded table by hours")

# FLAGS
df_hours["flag_shabbat"] = (
    (df_hours["date"].dt.dayofweek == 5) & (df_hours["hour"] < 18)
) | ((df_hours["date"].dt.dayofweek == 4) & (df_hours["hour"] >= 18))
df_hours["flag_weekend"] = df_hours["date"].dt.weekday.isin([4, 5])
df_hours["flag_night"] = (df_hours["hour"] <= 8) | (df_hours["hour"] >= 22)
df_hours["flag_work_hour"] = (~df_hours["flag_weekend"]) & (
    df_hours["hour"].apply(lambda x: (x <= 16) & (x >= 11))
)
df_hours["flag_home_hour"] = df_hours["flag_night"] | df_hours["flag_weekend"]

# nighttime geohashes
geohash_ident_night = (
    df_hours[(df_hours.hour <= 8) | (df_hours.hour >= 22)]
    .groupby(["identifier", "geohash"])["date"]
    .agg(["count", "nunique"])
)
# keep only users with at least 2 dwells (redundant!)
geohash_ident_night = geohash_ident_night[
    geohash_ident_night["count"] > 1

]  # if it is still noisy then we use ninique
geohash_ident_night = geohash_ident_night.reset_index()
geohash_ident_night["rank_geohash"] = geohash_ident_night.groupby(["identifier"])[
    "count"
].rank(ascending=False, method="first")


# shabbat geohashes
user_geohash_shbt = (
    df_hours[df_hours["flag_shabbat"]]
    .groupby(["identifier", "geohash"])["hour"]
    .count()
    .to_frame()
)

user_geohash_shbt["rank"] = user_geohash_shbt.groupby(["identifier"])["hour"].rank(
    ascending=False, method="first"
)

# outer join of shabbat and night geohashes
geohash_ident_night_shbt = geohash_ident_night.merge(
    user_geohash_shbt, on=["identifier", "geohash"], how="outer"
)
geohash_ident_night_shbt = geohash_ident_night_shbt.fillna(0)

geohash_ident_night_shbt.columns = [
    "identifier",
    "geohash",
    "night_hours_count",
    "nights_count",
    "rank_night",
    "shabbat_hours_counts",
    "rank_shabbat",
]
geohash_ident_night_shbt["weight"] = (
    geohash_ident_night_shbt["night_hours_count"]
    + geohash_ident_night_shbt["shabbat_hours_counts"]
)
geohash_ident_night_shbt["rank_weighted"] = geohash_ident_night_shbt.groupby(
    "identifier"
)["weight"].rank(ascending=False, method="first")

# home geohashes
geohash_ident_night_shbt_homes = geohash_ident_night_shbt[
    (geohash_ident_night_shbt.rank_weighted == 1)  # first from two ranks
    & (geohash_ident_night_shbt.shabbat_hours_counts >= 1)  # new filter
]
if geohash_ident_night_shbt_homes.empty:
    raise ValueError("No home locations identified after filtering")
dict_key_metrics["num_homes"] = geohash_ident_night_shbt_homes.shape[0]
print(f"Selected {dict_key_metrics['num_homes']} home locations")

# aggregate by geohash and add geometry
geohash_homes = (
    geohash_ident_night_shbt_homes.groupby(["geohash"])["identifier"]
    .nunique()
    .reset_index()
)
geohash_homes["geometry"] = pd.Series(
    geohash_homes.geohash.apply(lambda x: utils.geohash_to_polygon(x)),
    index=geohash_homes.index,
)

geohash_homes = utils.make_gdf(geohash_homes, geometry="geometry")

# Add stat area

stat_areas = gpd.read_file(
    utils.get_path("processed", "census", "census_gushdan_main_features.geojson")
)
stat_areas = stat_areas.to_crs(4326)

# merge
geohash_homes_stat_area = stat_areas[["YISHUV_STAT_2022", "geometry"]].sjoin(
    geohash_homes, predicate="intersects", how="inner"
)

users_stat_area = (
    geohash_homes_stat_area.groupby("YISHUV_STAT_2022")["identifier"]
    .sum()
    .reset_index()
)

# calculate pop by stat area
homes_stat_area = stat_areas[
    ["YISHUV_STAT_2022", "SHEM_YISHUV_ENG", "pop_approx", "WrkOutLoc_pcnt","geometry"]
].merge(users_stat_area, on="YISHUV_STAT_2022", how="left")
homes_stat_area = homes_stat_area.fillna(0)

homes_stat_area = homes_stat_area.reset_index()
homes_stat_area = utils.make_gdf(homes_stat_area, geometry="geometry")

corr_table = homes_stat_area.loc[
    homes_stat_area["SHEM_YISHUV_ENG"] != "BENE BERAQ", ["pop_approx", "identifier"]
].corr("spearman")

dict_key_metrics["home_pop_corr"] = corr_table.iloc[0, 1]

# Add spatial variance

df_filtered_dwells_with_homes = df_filtered_dwells.merge(
    geohash_ident_night_shbt_homes[["identifier", "geohash"]],
    on=["identifier"],
    suffixes=("", "_home"),
)
# add geometry columns
df_filtered_dwells_with_homes["geometry"] = pd.Series(
    df_filtered_dwells_with_homes.geohash.apply(
        lambda x: utils.geohash_to_polygon(x)
    ),
    index=df_filtered_dwells_with_homes.index,
)
df_filtered_dwells_with_homes["geometry_home"] = pd.Series(
    df_filtered_dwells_with_homes.geohash_home.apply(
        lambda x: utils.geohash_to_polygon(x)
    ),
    index=df_filtered_dwells_with_homes.index,
)
# TODO: add reading crs from settings
df_filtered_dwells_with_homes["geometry"] = gpd.GeoSeries(
    df_filtered_dwells_with_homes["geometry"], crs=4326
).to_crs(2039)
df_filtered_dwells_with_homes["geometry_home"] = gpd.GeoSeries(
    df_filtered_dwells_with_homes["geometry_home"], crs=4326
).to_crs(2039)
user_spatial_variance = (
    df_filtered_dwells_with_homes.groupby("identifier")
    .apply(utils.spatial_variance)
    .reset_index(name="spatial_variance")
)
dict_key_metrics["spatial_variance"] = user_spatial_variance[
    "spatial_variance"
].median()

# Merge work, home and weekends hours
geohash_user_home_hour = geohash_ident_night_shbt_homes[
    ["identifier", "geohash"]
].merge(
    df_hours[
        [
            "identifier",
            "geohash",
            "hour",
            "flag_weekend",
            "flag_home_hour",
            "flag_work_hour",
            "date",
        ]
    ],
    how="inner",
    on=["identifier", "geohash"],
)
# work hours at home- ( try different metrics because we dont know which one will work best for remote working)
user_home_days = (
    df_hours[df_hours["flag_home_hour"]]
    .groupby(["identifier"])["date"]
    .nunique()
    .rename("date_home_all")
    .reset_index()
)
# hours
user_home_work_hours = (
    geohash_user_home_hour[geohash_user_home_hour["flag_work_hour"]]
    .groupby(["identifier", "geohash"])["hour"]
    .count()
    .rename("hour_work_home")
)
# days
work_hours_home_day = (
    geohash_user_home_hour[
        (geohash_user_home_hour["flag_work_hour"])
    ]
    .groupby(["identifier", "geohash", "date"])
    .size()
    .reset_index(name="count")
)
user_home_work_days = (
    work_hours_home_day.groupby(["identifier", "geohash"])["date"]
    .nunique()
    .rename("date_work_home")
    .reset_index()
)
# total work hours
user_work_hours = (
    df_hours[df_hours["flag_work_hour"]]
    .groupby(["identifier"])
    .size()
    .rename("hour_work_all")
    .reset_index()
)
user_work_days = (
    df_hours[df_hours["flag_work_hour"]]
    .groupby(["identifier"])["date"]
    .nunique()
    .rename("date_work_all")
    .reset_index()
)

# total hours at home
user_home_total_hours = (
    geohash_user_home_hour.groupby(["identifier", "geohash"])["hour"]
    .count()
    .rename("hour_home_all")
    .to_frame()
)
geohash_ident_night_shbt_homes = (
    geohash_ident_night_shbt_homes.merge(
        user_home_work_hours, on=["identifier", "geohash"], how="left"
    )
    .merge(user_home_days, on=["identifier"], how="left")
    .merge(user_home_total_hours, on=["identifier", "geohash"], how="left")
    .merge(user_home_work_days, on=["identifier", "geohash"], how="left")
    .merge(user_work_hours, on=["identifier"], how="left")
    .merge(user_work_days, on=["identifier"], how="left")
).fillna(0)

# Key metrics
geohash_ident_night_shbt_homes["share_work_in_home_hours"] = (
    geohash_ident_night_shbt_homes["hour_work_home"]
    / geohash_ident_night_shbt_homes["hour_home_all"]
)
geohash_ident_night_shbt_homes["share_home_work_hours"] = np.where(
    geohash_ident_night_shbt_homes["hour_work_all"] == 0,
    None,
    geohash_ident_night_shbt_homes["hour_work_home"]
    / geohash_ident_night_shbt_homes["hour_work_all"],
)
geohash_ident_night_shbt_homes["share_work_in_home_days"] = (
    geohash_ident_night_shbt_homes["date_work_home"]
    / geohash_ident_night_shbt_homes["date_home_all"]
)
geohash_ident_night_shbt_homes["share_home_work_days"] = np.where(
    geohash_ident_night_shbt_homes["date_work_all"] == 0,
    None,
    geohash_ident_night_shbt_homes["date_work_home"]
    / geohash_ident_night_shbt_homes["date_work_all"],
)
geohash_ident_night_shbt_homes["share_home_hours_home"] = (
    geohash_ident_night_shbt_homes["night_hours_count"]
    + geohash_ident_night_shbt_homes["shabbat_hours_counts"]
) / geohash_ident_night_shbt_homes["hour_home_all"]
dict_shares = (
    geohash_ident_night_shbt_homes[
        [
            "share_work_in_home_hours",
            "share_home_work_hours",
            "share_work_in_home_days",
            "share_home_work_days",
            "share_home_hours_home",
        ]
    ]
    .mean()
    .to_dict()
)
dict_key_metrics = {**dict_key_metrics, **dict_shares}

# Run hours freq calculation

df_user_freq = calculate_frequencies(df_hours)
if df_user_freq.empty:
    raise ValueError("No frequency data calculated")

list_user_freq = df_user_freq[
    (df_user_freq.freq_work_hours_count >= 2)
    & (df_user_freq.freq_home_hours_count >= 2)
]["identifier"]
if len(list_user_freq) == 0:
    raise ValueError("No frequent users found after filtering")
geohash_ident_night_shbt_homes["flag_frequent_user"] = (
    geohash_ident_night_shbt_homes.identifier.isin(list_user_freq)
)
dict_key_metrics["freq_users"] = len(list_user_freq)
print(f"Found {dict_key_metrics['freq_users']} frequent users")

print("start saving results")
# Save results
save_dir = utils.get_path("processed", "dwells")

homes_stat_area.to_file(
    os.path.join(save_dir, f"home/home_stat_areas_night_shabbat_{MONTH}.geojson")
)
geohash_ident_night_shbt_homes.to_csv(
    os.path.join(save_dir, f"home/home_geohashes_night_shabbat_{MONTH}.csv")
)

with open(os.path.join(save_dir, "logs_homes_new.txt"), "a") as file:
    file.write(
        f"""Month: {MONTH}, num_homes:{dict_key_metrics["num_homes"]},
       corr_pop_homes:{dict_key_metrics['home_pop_corr']},
      spatial_variance:{dict_key_metrics['spatial_variance']},
        share home dwells in work hours:{dict_key_metrics['share_home_work_hours']}, 
        share remote work of all home_days:{dict_key_metrics["share_work_in_home_days"]}, 
        share remote work of all work days:{dict_key_metrics["share_home_work_days"]},
        share home hours in home dwells:{dict_key_metrics["share_home_hours_home"]},
        frequent users:{dict_key_metrics["freq_users"]}\n"""
    )
