import os
import pandas as pd
import geopandas as gpd
import argparse
from geofunctions import utils

"""
This script processes dwelling (location) data to identify work locations and analyze commuting patterns.

The script performs the following steps:
1. Reads filtered dwelling data and home location data for a specific month
2. Processes dwells for users with known home locations
3. Calculates distances between home and dwelling locations
4. Identifies work locations based on several criteria:
   - Minimum 3 hours and 2 days per month
   - Located further than 1km from home
   - No weekend signals
   - Highest frequency ranking among remaining locations
5. Creates a user dashboard with commuting statistics
6. Adds statistical area information and calculates commute metrics by statistical area
"""


# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Read a Parquet file with a specified month filter."
)
parser.add_argument(
    "--month", type=str, default=1, help="Specify the number of files to process"
)
args = parser.parse_args()
MONTH = args.month

def load_data(month):
    """Load and prepare the initial dwelling and home location datasets.
    
    Args:
        month (str): Month identifier (e.g. '202301')
        
    Returns:
        tuple: (filtered_dwells DataFrame, homes DataFrame)
    """
    df_filtered_good = pd.read_parquet(
        utils.get_path("raw", "dwells", f"data_filtered_{month}.parquet")
    )
    df_filtered_dwells = utils.make_gdf(df_filtered_good, geometry="the_geom")
    
    path_processed_dir = utils.get_path("processed", "dwells", "home") 
    homefile = f"home_geohashes_night_shabbat_{month}.csv"
    df_homes = pd.read_csv(os.path.join(path_processed_dir, homefile), index_col=0)
    
    return df_filtered_dwells, df_homes

def process_dwelling_hours(df_hours):
    """Process hourly dwelling data to identify potential work locations.
    
    Args:
        df_hours (pd.DataFrame): Hourly dwelling data
        
    Returns:
        pd.DataFrame: Processed dwelling data with hour counts and days counts
    """
    return (
        df_hours[~df_hours["flag_home_hour"]]
        .groupby(["identifier", "geohash"])
        .agg({
            "identifier": "size",  # for hours count
            "date": "nunique"      # for days count
        })
        .rename(columns={"identifier": "hours_count", "date": "days_count"})
        .reset_index()
    )

def filter_work_locations(users_geohash, min_hours=3, min_days=2, home_distance_threshold=1000):
    """Filter locations based on work location criteria.
    
    Args:
        users_geohash (pd.DataFrame): User-location data
        min_hours (int): Minimum hours required
        min_days (int): Minimum days required
        home_distance_threshold (int): Minimum distance from home in meters
        
    Returns:
        pd.DataFrame: Filtered work locations
    """
    # Apply minimum hours and days filters
    filtered = users_geohash[
        (users_geohash.hours_count >= min_hours) & 
        (users_geohash.days_count >= min_days)
    ]
    
    # Filter by distance and weekend signals
    filtered = filtered[filtered["distance_home"] > home_distance_threshold]
    
    # Select highest frequency location
    filtered["rank_hours"] = filtered.groupby("identifier")["hours_count"].rank(
        ascending=False, method="first"
    )
    return filtered[filtered.rank_hours == 1]

def identify_work_locations(df_hours, min_hours=3, min_days=2, home_distance_threshold=1000):
    """Identify work locations based on dwelling patterns.
    
    Args:
        df_hours (pd.DataFrame): Hourly dwelling data
        min_hours (int): Minimum hours required at location
        min_days (int): Minimum days required at location
        home_distance_threshold (int): Minimum distance from home in meters
        
    Returns:
        pd.DataFrame: Work locations for each user
    """
    users_geohash = process_dwelling_hours(df_hours)
    users_geohash = filter_work_locations(
        users_geohash, 
        min_hours=min_hours,
        min_days=min_days,
        home_distance_threshold=home_distance_threshold
    )
    return users_geohash

def calculate_commute_metrics(df_user_dashboard):
    """Calculate commuting statistics by statistical area.
    
    Args:
        df_user_dashboard (pd.DataFrame): User dashboard with work/home locations
        
    Returns:
        gpd.GeoDataFrame: Statistical areas with commute metrics
    """
    stat_areas = load_statistical_areas()
    gdf_user_dashboard = utils.make_gdf(df_user_dashboard, geometry="home_geometry", crs=4326)
    return process_commute_statistics(stat_areas, gdf_user_dashboard)

def process_user_dwells(df_filtered_dwells, df_homes):
    """Process dwelling data for users with identified home locations.
    
    Args:
        df_filtered_dwells (pd.DataFrame): Filtered dwelling data
        df_homes (pd.DataFrame): Home location data
        
    Returns:
        pd.DataFrame: Processed dwelling data with home location information
    """
    # Merge home data with dwells
    df_dwells_users_home = df_homes[
        ["identifier", "geohash", "night_hours_count", "nights_count"]
    ].merge(
        df_filtered_dwells, 
        on="identifier", 
        how="inner", 
        suffixes=("_home", "")
    )
    
    # Add date and weekend flags
    df_dwells_users_home["date"] = pd.to_datetime(df_dwells_users_home["date"])
    df_dwells_users_home["flag_weekend"] = df_dwells_users_home["date"].dt.weekday.isin([4, 5])
    
    # Calculate distances between home and current location
    df_dwells_users_home["home_geometry"] = df_dwells_users_home["geohash_home"].apply(
        utils.geohash_to_polygon
    )
    df_dwells_users_home["geometry"] = df_dwells_users_home["geohash"].apply(
        utils.geohash_to_polygon
    )
    df_dwells_users_home["distance_home"] = utils.calculate_distance(
        df_dwells_users_home, "home_geometry", "geometry", crs=4326
    )
    
    # Add home geohash flag
    df_dwells_users_home["flag_home_geohash"] = df_dwells_users_home["distance_home"] <= 250
    
    return df_dwells_users_home

# Explode the hours into separate rows
df_hours = df_dwells_users_home.explode("hours_in_interval")
df_hours["hour"] = df_hours["hours_in_interval"].dt.hour
df_hours["flag_night"] = (df_hours["hour"] <= 8) | (df_hours["hour"] >= 22)
df_hours["flag_home_hour"] = df_hours["flag_night"] | df_hours["flag_weekend"]

# Filters for work location:
# - Minimum 3 hours and 2 days per month
# - Located further than 1 km from home
# - Doesn't have weekend signals
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
    .reset_index()
)

# Minimum 3 hours and 2 days
users_geohash = users_geohash[
    (users_geohash.hours_count >= 3) & (users_geohash.days_count >= 2)
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
users_geohas_no_home = users_geohas_home[users_geohas_home["distance_home"] > 1000]

# Exclude locations with signals at weekend
no_weekend_signals = df_hours.groupby(["identifier", "geohash"])["flag_weekend"].sum()[
    df_hours.groupby(["identifier", "geohash"])["flag_weekend"].sum() == 0
]
users_geohas_no_home = users_geohas_no_home.merge(
    no_weekend_signals.reset_index(), on=["identifier", "geohash"], how="inner"
)

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
    how="inner",
    on=["identifier"],
    suffixes=("", "_work"),
)

df_dwells_users_home["geometry_work"] = df_dwells_users_home["geohash_work"].apply(
    utils.geohash_to_polygon
)
df_dwells_users_home["distance_work"] = utils.calculate_distance(
    df_dwells_users_home, "geometry_work", "geometry", crs=4326
)

df_dwells_users_home["flag_work_geohash"] = df_dwells_users_home["distance_work"] <= 250

# Create users dashboard

df_days_work_all = (
    df_dwells_users_home.groupby("identifier")["date"]
    .nunique()
    .to_frame()
    .join(
        df_dwells_users_home[df_dwells_users_home["flag_work_geohash"]]
        .groupby("identifier")["date"]
        .nunique()
        .rename("work_location_date")
        .to_frame(),
        how="inner",
    )
    .join(
        df_dwells_users_home[~df_dwells_users_home["flag_weekend"]]
        .groupby("identifier")["date"]
        .nunique()
        .rename("work_days_with_signals")
        .to_frame(),
        how="inner",
    )
)
df_days_work_all["share_days_work_location"] = (
    df_days_work_all["work_location_date"] / df_days_work_all["date"]
)
df_days_work_all["share_workdays_work_location"] = (
    df_days_work_all["work_location_date"] / df_days_work_all["work_days_with_signals"]
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
df_user_dashbord["weighted_commuting_distance"] = (
    df_user_dashbord["share_days_work_location"] * df_user_dashbord["distance_home"]
)
df_user_dashbord = df_user_dashbord.rename(
    columns={"distance_home": "commuting_distance"}
)
df_user_dashbord.to_csv(
    utils.get_path("processed", "dwells", f"work/users_work_home_{MONTH}.csv")
)
print(f"users_work_home_{MONTH}.csv saved")


# Add stat area

stat_areas = gpd.read_file(
    utils.get_path("processed", "census", "census_gushdan_main_features.geojson")
)
stat_areas = stat_areas.to_crs(4326)
gdf_user_dashbord = utils.make_gdf(df_user_dashbord, geometry="home_geometry", crs=4326)

# merge
user_dashbord_stat_area = stat_areas[["YISHUV_STAT_2022", "geometry"]].sjoin(
    gdf_user_dashbord, predicate="intersects", how="inner"
)

stat_area_info = (
    user_dashbord_stat_area.groupby("YISHUV_STAT_2022")[
        ["commuting_distance", "weighted_commuting_distance"]
    ]
    .mean()
    .join(
        user_dashbord_stat_area.groupby("YISHUV_STAT_2022")["identifier"]
        .size()
        .to_frame()
    )
)

# Calculate commute metrics bt stat area
stat_areas_commute_data = stat_areas[
    ["YISHUV_STAT_2022", "SHEM_YISHUV_ENG", "geometry"]
].merge(stat_area_info.reset_index(), on=["YISHUV_STAT_2022"])
stat_areas_commute_data["month"] = MONTH
stat_areas_commute_data.to_file(
    utils.get_path(
        "processed", "dwells", f"data_statistics/stat_area_commuting_{MONTH}.geojson"
    )
)
