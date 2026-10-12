import os
import geopandas as gpd
import pandas as pd
import argparse
from geofunctions import utils
import holidays
from datetime import timedelta

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Read a Parquet file with a specified month filter."
)
parser.add_argument(
    "--month", type=str, default=1, help="Specify the number of files to process"
)
args = parser.parse_args()
MONTH = args.month
MIN_DIST_THRESHOLD = 1000
MIN_WORK_DAYS = 2
MIN_HOURS = 3
MIN_HOURS_ONE_DAY = 2
# Read files

df_filtered_good = pd.read_parquet(
    utils.get_path("raw", "dwells", f"data_filtered_{MONTH}.parquet")
)
df_filtered_dwells = utils.make_gdf(df_filtered_good, geometry="the_geom")

path_processed_dir = utils.get_path("processed", "dwells", "home")
homefile = f"home_geohashes_night_shabbat_{MONTH}.csv"
df_homes = pd.read_csv(os.path.join(path_processed_dir, homefile), index_col=0)


# Keep only dwells of users with clear home location
df_dwells_users_home = df_homes[
    ["identifier", "geohash", "night_hours_count", "nights_count"]
].merge(df_filtered_dwells, on="identifier", how="inner", suffixes=("_home", ""))

df_dwells_users_home["date"] = pd.to_datetime(df_dwells_users_home["date"])

# Define weekend

# Define Israeli holidays
israel_holidays = holidays.country_holidays("IL", years=MONTH[:4])
one_day_before_holidays = []
one_day_before_holidays = one_day_before_holidays + [
    date - timedelta(days=1) for date in israel_holidays.keys()
]
# one_day_before_holidays_str = [
#     date.strftime("%Y-%m-%d") for date in one_day_before_holidays
# ]
# Ensure 'date' is in datetime format first
df_dwells_users_home["date"] = pd.to_datetime(df_dwells_users_home["date"])

election_days = ["2019-04-09", "2019-09-17", "2021-03-23", "2022-11-01", "2024-02-27"]
# Create the 'flag_weekend' column
df_dwells_users_home["flag_weekend"] = (
    df_dwells_users_home["date"].dt.weekday.isin([4, 5])
    | df_dwells_users_home["date"].dt.date.isin(  # Check if holidays
        israel_holidays.keys()
    )
    | df_dwells_users_home["date"].dt.date.isin(  # Check if erev hug
        one_day_before_holidays
    )
    | df_dwells_users_home["date"].dt.date.isin(pd.to_datetime(election_days).date)  #
)  # Check if elections day # Modified on 8/02/2025


### Add distance between home and geohash
df_dwells_users_home["home_geometry"] = df_dwells_users_home["geohash_home"].apply(
    utils.geohash_to_polygon
)
df_dwells_users_home["geometry"] = df_dwells_users_home["geohash"].apply(
    utils.geohash_to_polygon
)
df_dwells_users_home["distance_home"] = utils.calculate_distance(
    df_dwells_users_home, "home_geometry", "geometry", crs=4326
)

# Identifying work location

df_dwells_users_home["flag_home_geohash"] = df_dwells_users_home["distance_home"] <= 250


### explode dwells to hourly signals
df_dwells_users_home["hours_in_interval"] = df_dwells_users_home.apply(
    lambda row: pd.date_range(
        start=row["start_date_time"].floor("h"),
        end=row["end_date_time"].ceil("h") - pd.Timedelta(hours=1),
        freq="h",
    ).tolist(),
    axis=1,
)

# Explode the hours into separate rows
file_path = utils.get_path(
    "processed", "dwells", f"samples/df_hours_{MONTH}.csv"
)  # Replace with your file path

if os.path.exists(file_path):
    print("file found, start uploading")
    df_hours = pd.read_csv(
        utils.get_path("processed", "dwells", f"samples/df_hours_{MONTH}.csv")
    )
    df_hours = df_hours[["identifier", "hour", "timestamp", "geohash"]]
    df_hours["timestamp"] = pd.to_datetime(df_hours["timestamp"])
    df_hours = df_dwells_users_home.merge(df_hours, on=["identifier", "timestamp", "geohash"])
    print("Uploaded df hours from existing file")
else:

    df_hours = df_dwells_users_home.explode("hours_in_interval")
    df_hours["hour"] = df_hours["hours_in_interval"].dt.hour
    print("Exploded the hours into separate rows")
    df_hours.to_csv(
        utils.get_path("processed", "dwells", f"samples/df_hours_{MONTH}.csv")
    )
    print("Saved exploded hours into separate rows")
df_hours["flag_night"] = (df_hours["hour"] <= 8) | (df_hours["hour"] >= 22)
df_hours["flag_home_hour"] = df_hours["flag_night"] | df_hours["flag_weekend"]

# Filters for work location:
# - Minimum 3 hours and 2 days per month
# - Minimum one time with at least 2 hours in one day
# - Located further than 1 km from home
# - Doesn't have Shabbat signals # updated
# - highest hours frequency ranking among left locations

users_geohash = (
    df_hours[~df_hours["flag_home_hour"]]
    .groupby(["identifier", "geohash"])
    .size()
    .rename("hours_count")
    .to_frame()
    .join(
        df_hours[~df_hours["flag_home_hour"]]
        .groupby(["identifier", "geohash"])["date"]
        .nunique()
        .rename("days_count")
        .to_frame()
    )
    .join(
        df_hours[~df_hours["flag_home_hour"]]
        .groupby(["identifier", "geohash", "date"])
        .size()
        .groupby(["identifier", "geohash"])
        .max()
        .rename("hours_max_day_count")
        .to_frame()
    )
    .reset_index()
)

# Minimum 3 hours and 3 days
users_geohash = users_geohash[
    (users_geohash.hours_count >= MIN_HOURS)
    & (users_geohash.days_count >= MIN_WORK_DAYS)
    & (users_geohash.hours_max_day_count >= MIN_HOURS_ONE_DAY)
]


users_geohas_home = users_geohash.merge(
    df_homes[["identifier", "geohash", "rank_night"]],
    how="left",
    on=["identifier", "geohash"],
)

# Exclude locations near home
users_geohas_home = users_geohas_home.merge(
    df_dwells_users_home[["identifier", "geohash", "distance_home"]].drop_duplicates(),
    on=["identifier", "geohash"],
)
users_geohas_no_home = users_geohas_home[
    users_geohas_home["distance_home"] > MIN_DIST_THRESHOLD
]

# Exclude locations with signals at Shabbat and holidays
df_hours["flag_shabbat"] = (
    (df_hours["date"].dt.dayofweek == 5) & (df_hours["hour"] < 18)
) | ((df_hours["date"].dt.dayofweek == 4) & (df_hours["hour"] >= 18))
no_weekend_signals = df_hours.groupby(["identifier", "geohash"])["flag_shabbat"].sum()[
    df_hours.groupby(["identifier", "geohash"])["flag_shabbat"].sum() == 0
]
holidays_signals = df_hours.groupby(["identifier", "geohash"])["date"].apply(
    lambda x: sum(x.dt.date.isin(israel_holidays.keys()))  # Check if holidays
)
no_holidays_signals = holidays_signals[holidays_signals == 0]
users_geohas_no_home = users_geohas_no_home.merge(
    no_weekend_signals.reset_index(), on=["identifier", "geohash"], how="inner"
).merge(no_holidays_signals.reset_index(), on=["identifier", "geohash"], how="inner")

# Calculate rank
users_geohas_no_home["rank_hours_out_home"] = users_geohas_no_home.groupby(
    "identifier"
)["hours_count"].rank(ascending=False, method="first")

# Select most frequent place
users_geohash_work = users_geohas_no_home[
    (users_geohas_no_home.rank_hours_out_home == 1)
]

# Add the flag of work location to dwells dataset
df_dwells_users_home = df_dwells_users_home.merge(
    users_geohash_work[["identifier", "geohash"]],
    how="left",
    on=["identifier"],
    suffixes=("", "_work"),
)

df_dwells_users_home["flag_work_geohash"] = (
    df_dwells_users_home["geohash"] == df_dwells_users_home["geohash_work"]
)

# Create users dashboard

df_days_work_all = (
    df_dwells_users_home.groupby("identifier")["date"]
    .nunique()
    .to_frame()
    .join(
        df_dwells_users_home[df_dwells_users_home["flag_work_geohash"]]
        .groupby("identifier")["date"]
        .nunique()
        .rename("work_date")
        .to_frame(),
        how="inner",
    )
)
df_days_work_all["share_days_work_location"] = (
    df_days_work_all["work_date"] / df_days_work_all["date"]
)
df_days_work_all = df_days_work_all.merge(
    df_dwells_users_home[df_dwells_users_home.flag_work_geohash][
        ["identifier", "distance_home", "home_geometry"]
    ].drop_duplicates(),
    on="identifier",
    how="left",
)
df_user_dashbord = df_days_work_all.merge(df_homes, on="identifier", how="inner")
df_user_dashbord = df_user_dashbord.merge(
    users_geohash_work[["identifier", "geohash"]],
    on=["identifier"],
    suffixes=("_home", "_work"),
    how="inner",
)
assert df_user_dashbord["identifier"].nunique() == len(df_user_dashbord)
print(f"Users with work:{df_user_dashbord["identifier"].nunique()}")
df_user_dashbord.to_csv(
    utils.get_path("processed", "dwells", f"work/users_work_home_{MONTH}.csv")
)
print(f"users_work_home_{MONTH}.csv saved")


# # Add stat area

# stat_areas = gpd.read_file(
#     utils.get_path("processed", "census", "census_gushdan_main_features.geojson")
# )
# stat_areas = stat_areas.to_crs(4326)
# gdf_user_dashbord = utils.make_gdf(df_user_dashbord, geometry="home_geometry", crs=4326)

# # merge
# user_dashbord_stat_area = stat_areas[["YISHUV_STAT_2022", "geometry"]].sjoin(
#     gdf_user_dashbord, predicate="intersects", how="inner"
# )

# stat_area_info = (
#     user_dashbord_stat_area.groupby("YISHUV_STAT_2022")[
#         ["commuting_distance", "weighted_commuting_distance"]
#     ]
#     .mean()
#     .join(
#         user_dashbord_stat_area.groupby("YISHUV_STAT_2022")["identifier"]
#         .size()
#         .to_frame()
#     )
# )

# # Calculate commute metrics bt stat area
# stat_areas_commute_data = stat_areas[
#     ["YISHUV_STAT_2022", "SHEM_YISHUV_ENG", "geometry"]
# ].merge(stat_area_info.reset_index(), on=["YISHUV_STAT_2022"])
# stat_areas_commute_data["month"] = MONTH
# stat_areas_commute_data.to_file(
#     utils.get_path("processed", "statistics", f"stat_area_commuting_{MONTH}.geojson")
# )
