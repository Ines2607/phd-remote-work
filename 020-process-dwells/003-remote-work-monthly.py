"""
Remote Work Analysis Script
--------------------------

This script processes location data to analyze remote work patterns by tracking user movements
between home, office, and other locations. It calculates various remote work metrics and 
generates visualizations of work patterns.

Processing Steps:
1. Data Loading and Initial Processing:
   - Loads filtered dwell data from Parquet files
   - Loads work/home location data from CSV
   - Merges dwell data with work/home locations
   - Converts geometries to GeoDataFrame format

2. Location Classification:
   - Calculates distances between dwells and work/home locations using geohashing
   - Classifies each dwell as 'home', 'work', or 'other' based on 250m distance threshold
   - Flags weekend days and holidays using Israeli calendar

3. Temporal Processing:
   - Explodes dwell intervals into hourly observations
   - Handles overlapping dwells by removing duplicates
   - Creates hour-of-day features for analysis

4. Probability Calculations:
   - Calculates base probabilities of being at each location type
   - Computes conditional probabilities for each hour
   - Applies smoothing to probabilities using location weights
   - Determines most likely location type for each hour

5. Remote Work Detection:
   - Identifies work hours based on probability patterns
   - Flags remote work using three different approaches:
     a. Basic: Home location during work hours on non-office days
     b. Smooth probability: Using smoothed location probabilities
     c. Minimum work probability: Using work probability threshold

6. Metric Calculation:
   - Aggregates remote work flags to daily level
   - Calculates user-level statistics:
     * Total work days
     * Remote work days (all three approaches)
     * Remote work shares
   - Generates hourly statistics for visualization

7. Output Generation:
   - Saves user-level remote work metrics to CSV
   - Saves hourly statistics to CSV
   - Creates visualization plots:
     * Location patterns by hour for office days
     * Location patterns by hour for all workdays
     * Separate plots for Android and iPhone users

Input Data Requirements:
- Filtered dwell data (Parquet format):
  * identifier: Unique user ID
  * start_date_time: Start time of dwell
  * end_date_time: End time of dwell
  * the_geom: Geographic coordinates
  * identifier_type: Device type (GAID/IDFA)

- Work/home locations (CSV format):
  * identifier: Unique user ID
  * geohash_home: Geohash of home location
  * geohash_work: Geohash of work location

Output Files:
1. user_remote_work_{MONTH}.csv:
   - User-level statistics including remote work days and shares
2. remote_work_stats_hour_date_{MONTH}.csv:
   - Hourly remote work statistics
3. plots/work_home_hours_{MONTH}.png:
   - Visualization of daily patterns

Usage:
    python 003-remote-work-monthly.py --month YYYYMM

Dependencies:
    - pandas: Data processing and analysis
    - geopandas: Geographic data handling
    - matplotlib: Basic plotting
    - seaborn: Advanced visualization
    - holidays: Holiday calendar handling
    - argparse: Command line argument parsing
    - custom utils module with functions:
      * get_path: Path handling
      * make_gdf: GeoDataFrame creation
      * geohash_to_polygon: Geohash conversion
      * calculate_distance: Distance calculations

Author: [Your Name]
Last Updated: [Date]
Version: 1.0
"""

from datetime import timedelta
import os
import pandas as pd
import argparse
import sys
import holidays
from matplotlib import pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = "/Users/inessat/Documents/phd/empirics/israel/notebooks/phd-remote-work"
if project_root not in sys.path:
    sys.path.append(project_root)
from geofunctions import utils


def return_prob_a(location_type, df_hours):
    if location_type == "work":
        name = "prob_office"
    elif location_type == "home":
        name = "prob_home"
    else:
        name = "prob_3rdplace"
    prob_a = (
        df_hours[df_hours.office_day]
        .groupby(["identifier", "month"])[f"flag_{location_type}_geohash"]
        .mean()
        .rename(name)
        .to_frame()
    )
    return prob_a


def return_prob_ba(location_type, df_hours):
    if location_type == "work":
        name = "prob_hour_at_office"
    elif location_type == "home":
        name = "prob_hour_at_home"
    else:
        name = "prob_hour_at_3rdplace"
    prob_ba = (
        (
            df_hours[
                (df_hours.office_day) & (df_hours[f"flag_{location_type}_geohash"])
            ]
            .groupby(["identifier", "month", "hour"])
            .size()
            / df_hours[
                (df_hours.office_day) & (df_hours[f"flag_{location_type}_geohash"])
            ]
            .groupby(["identifier", "month"])
            .size()
        )
        .rename(name)
        .to_frame()
    )
    return prob_ba


def conditional_probability_table(df_hours):
    prob_a_work = return_prob_a("work", df_hours)
    prob_a_home = return_prob_a("home", df_hours)
    prob_a_other = return_prob_a("other", df_hours)
    prob_b = (
        (
            df_hours[df_hours.office_day]
            .groupby(["identifier", "month", "hour"])
            .size()
            / df_hours[df_hours.office_day].groupby(["identifier", "month"]).size()
        )
        .rename("prob_hour")
        .to_frame()
    )
    prob_ba_work = return_prob_ba("work", df_hours)
    prob_ba_home = return_prob_ba("home", df_hours)
    prob_ba_other = return_prob_ba("other", df_hours)

    df_prob = (
        prob_b.join(prob_a_work, how="left")
        .join(prob_a_home, how="left")
        .join(prob_a_other, how="left")
        .join(prob_ba_work, how="left")
        .join(prob_ba_home, how="left")
        .join(prob_ba_other, how="left")
        .fillna(0)
    )
    df_prob = df_prob.reset_index()

    # Conditional probability
    df_prob["prob_office_at_hour"] = (
        df_prob["prob_hour_at_office"] * df_prob["prob_office"] / df_prob["prob_hour"]
    )
    df_prob["prob_home_at_hour"] = (
        df_prob["prob_hour_at_home"] * df_prob["prob_home"] / df_prob["prob_hour"]
    )

    # Modified 19/01. compare with average probability
    # df_prob["expected_prob_office_hour"] = df_prob["prob_office_at_hour"].mean()
    # df_prob["expected_prob_home_hour"] = df_prob["prob_home_at_hour"].mean()

    df_prob["prob_3rdplace_at_hour"] = (
        df_prob["prob_hour_at_3rdplace"]
        * df_prob["prob_3rdplace"]
        / df_prob["prob_hour"]
    )

    df_prob["flag_weekend"] = False

    return df_prob


def process_month_remote_work(df_hours: pd.DataFrame) -> pd.DataFrame:
    """
    Process remote work data for a given DataFrame. Calculates probabilities
    and returns processed DataFrame with remote work metrics.
    Now limits remote work detection to hours between 8:00-20:00.

    Args:
        df_hours (pd.DataFrame): DataFrame containing hourly location data

    Returns:
        pd.DataFrame: Processed DataFrame with remote work metrics
    """

    if (
        "prob_office_at_hour" not in df_hours.columns
        or "prob_home_at_hour" not in df_hours.columns
    ):
        raise ValueError(
            "Missing probability columns. Run conditional_probability_table first."
        )

    df_hours["prob_3rdplace_at_hour"] = 1 - (
        df_hours["prob_office_at_hour"] + df_hours["prob_home_at_hour"]
    )

    # Calculate general probabilities for office days

    if len(df_hours[(df_hours["office_day"]) & (~df_hours.flag_weekend)]) == 0:
        raise ValueError("No data for office days found")

    general_probability = pd.pivot_table(
        df_hours[(df_hours["office_day"]) & (~df_hours.flag_weekend)],
        index="hour",
        columns="location_type",
        values="flag_work_hours",  # Bug fix 4: Specify values column
        aggfunc="count",
        fill_value=0,
    ).div(
        df_hours[(df_hours["office_day"]) & (~df_hours.flag_weekend)]
        .groupby("hour")
        .size(),
        axis=0,
    )

    df_hours = df_hours.merge(
        general_probability.reset_index(), on="hour", how="left"
    ).fillna(0)

    required_columns = ["office_day", "location_type", "work", "home", "other"]
    missing_columns = [col for col in required_columns if col not in df_hours.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create predicted location type
    location_prob_columns = [
        "prob_home_at_hour",
        "prob_office_at_hour",
        "prob_3rdplace_at_hour",
    ]
    df_hours["predicted_location_type"] = df_hours[location_prob_columns].idxmax(axis=1)

    df_hours["predicted_location_type"] = df_hours["predicted_location_type"].map(
        {
            "prob_home_at_hour": "home",
            "prob_office_at_hour": "work",
            "prob_3rdplace_at_hour": "other",
        }
    )

    # Calculate work hours and remote work flags
    df_hours["is_working_hours"] = (df_hours["hour"] >= 8) & (df_hours["hour"] <= 19)
    # Modified on 5th of February
    df_hours["flag_work_hours"] = (df_hours["predicted_location_type"] == "work") & (
        df_hours["is_working_hours"]
    )
    df_hours["flag_work_hours_day"] = (
        df_hours.groupby(["identifier", "date"])["flag_work_hours"].transform("sum") > 0
    )
    # Approach 1: Calculate remote work flag as being at home location in work hours on non-office days
    # Add working hours flag

    # Modify remote work flags to only consider working hours
    df_hours["flag_remote_work"] = (
        (df_hours["location_type"] == "home")
        & (df_hours["flag_work_hours"])
        & (~df_hours["office_day"])
        & (df_hours["is_working_hours"])  # New condition
    )
    # Approach 2: Calculate remote work flag as being at home location in work hours ( based on smoothed probability) on non-office days

    # Calculate smooth probabilities
    df_hours["smoothed_prob_office"] = (
        df_hours["prob_office_at_hour"] * df_hours["work"]
    )
    df_hours["smoothed_prob_home"] = df_hours["prob_home_at_hour"] * df_hours["home"]
    df_hours["smoothed_prob_3rdplace"] = (
        df_hours["prob_3rdplace_at_hour"] * df_hours["other"]
    )

    smooth_prob_columns = [
        "smoothed_prob_home",
        "smoothed_prob_office",
        "smoothed_prob_3rdplace",
    ]
    df_hours["predicted_location_type_smooth_prob"] = df_hours[
        smooth_prob_columns
    ].idxmax(axis=1)

    df_hours["flag_remote_work_smooth_prob"] = (
        (df_hours["predicted_location_type_smooth_prob"] == "smoothed_prob_office")
        & (df_hours["location_type"] == "home")
        & (df_hours["is_working_hours"])  # New condition. 08/02/2025
        & (1 - df_hours["office_day"])
    )

    # Approach 3: Calculate minimum work probability flag
    df_hours["flag_remote_work_min_work_prob"] = (
        (df_hours["location_type"] == "home")
        & (df_hours["flag_work_hours"])
        & (df_hours["work"] > 0.5)
        & (df_hours["is_working_hours"])  # New condition
        & (1 - df_hours["office_day"])  # New condition. 08/02/2025
    )
    try:
        df_hours_remote_work = (
            df_hours[df_hours["flag_work_hours_day"]]
            .groupby(["identifier", "identifier_type", "date"])[
                [
                    "flag_remote_work",
                    "flag_remote_work_smooth_prob",
                    "flag_remote_work_min_work_prob",
                ]
            ]
            .agg(lambda x: sum(x) >= 1)
            .reset_index()
        )
    except KeyError as e:
        raise KeyError(f"Missing column in groupby operation: {e}")

    # Approach 4: Smooth day flag instead of hour probability
    df_hours["flag_remote_work_smoothed"] = (
        df_hours["flag_remote_work"] * df_hours["work"]
    )  # New from 8/02/2025

    # Approach 5: Keep smooth prob instead of flag
    df_hours["remote_work_smooth_prob"] = (
        df_hours["smoothed_prob_office"]
        * df_hours["flag_home_geohash"]
        * (1 - df_hours["office_day"])
    )
    # Calculate user-level statistics
    df_user_days_work_remote = (
        df_hours_remote_work.groupby(["identifier", "identifier_type"])
        .agg(
            {
                "flag_remote_work": ["sum", "count"],
                "flag_remote_work_smooth_prob": "sum",
                "flag_remote_work_min_work_prob": "sum",
            }
        )
        .reset_index()
    )
    metric_3 = (
        df_hours[df_hours["flag_work_hours_day"]]
        .groupby(["identifier", "date"])["flag_remote_work_smoothed"]
        .max()
        .groupby("identifier")
        .mean()
        .rename("flag_remote_work_smoothed_max")
    )
    metric_4 = (
        df_hours[df_hours["flag_work_hours_day"]]
        .groupby(["identifier", "date"])["flag_remote_work_smoothed"]
        .apply(lambda x: max(x) > 0.5)
        .groupby("identifier")
        .mean()
        .rename("flag_remote_work_smoothed_max_thresh")
    )

    df_user_days_work_remote.columns = [
        "identifier",
        "identifier_type",
        "remote_work_days",
        "total_work_days",
        "remote_work_days_smooth",
        "remote_work_days_min_work",
    ]

    df_user_days_work_remote = df_user_days_work_remote.merge(
        metric_3, on="identifier", how="left"
    ).merge(metric_4, on="identifier", how="left")

    # Calculate shares
    for metric in ["", "_smooth", "_min_work"]:
        df_user_days_work_remote[f"remote_work_share{metric}"] = (
            df_user_days_work_remote[f"remote_work_days{metric}"]
            / df_user_days_work_remote["total_work_days"]
        )

    return df_user_days_work_remote, df_hours


# def plot_remote_work_heatmap(df_hours: pd.DataFrame):
#     remote_work_stats_hour_date = (
#         df_hours.groupby(["date", "hour"])["flag_remote_work"].sum().reset_index()
#     )
#     sns.heatmap(
#         pd.pivot(
#             remote_work_stats_hour_date,
#             index="date",
#             columns="hour",
#             values="flag_remote_work",
#         ),
#         cmap="YlOrRd",
#     )
#     plt.show()


# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Read a Parquet file with a specified month filter."
)
parser.add_argument(
    "--month", type=str, default=1, help="Specify the number of files to process"
)
args = parser.parse_args()
MONTH = args.month

df_filtered_dwells = pd.read_parquet(
    utils.get_path("raw", "dwells", f"data_filtered_{MONTH}.parquet")
)

df_filtered_dwells = utils.make_gdf(df_filtered_dwells, geometry="the_geom")

df_freq_home_work = pd.read_csv(
    os.path.join(
        utils.get_path("processed", "dwells", "work"),
        f"users_work_home_{MONTH}.csv",
    ),
    index_col=0,
)
df_freq_home_work["month"] = MONTH
df_filtered_dwells["month"] = MONTH
df_freq_home_work = df_freq_home_work.drop_duplicates(["identifier"])

print("files are uploaded")
df_freq_full_home_work = df_filtered_dwells.merge(
    df_freq_home_work[["identifier", "geohash_home", "geohash_work", "month"]],
    on=["identifier", "month"],
)

# Add flags for work and home geohashes

df_freq_full_home_work["geometry_work"] = df_freq_full_home_work["geohash_work"].apply(
    utils.geohash_to_polygon
)
df_freq_full_home_work["distance_work"] = utils.calculate_distance(
    df_freq_full_home_work, "geometry_work", "the_geom", crs=4326
)

df_freq_full_home_work["flag_work_geohash"] = (
    df_freq_full_home_work["distance_work"] <= 250
)

df_freq_full_home_work["geometry_home"] = df_freq_full_home_work["geohash_home"].apply(
    utils.geohash_to_polygon
)
df_freq_full_home_work["distance_home"] = utils.calculate_distance(
    df_freq_full_home_work, "geometry_home", "the_geom", crs=4326
)

df_freq_full_home_work["flag_home_geohash"] = (
    df_freq_full_home_work["distance_home"] <= 250
)
df_freq_full_home_work["flag_other_geohash"] = (
    ~df_freq_full_home_work["flag_home_geohash"]
) & (~df_freq_full_home_work["flag_work_geohash"])

# define dwell type for  each user
df_freq_full_home_work["location_type"] = df_freq_full_home_work[
    ["flag_home_geohash", "flag_work_geohash"]
].apply(
    lambda r: (
        "home"
        if r["flag_home_geohash"]
        else ("work" if r["flag_work_geohash"] else "other")
    ),
    axis=1,
)


# explode dwells to hourly signals
df_freq_full_home_work["hours_in_interval"] = df_freq_full_home_work.apply(
    lambda row: pd.date_range(
        start=row["start_date_time"].floor("h"),
        end=row["end_date_time"].ceil("h") - pd.Timedelta(hours=1),
        freq="h",
    ).tolist(),
    axis=1,
)

# Explode the hours into separate rows
df_hours = df_freq_full_home_work.explode("hours_in_interval")
df_hours["hour"] = df_hours["hours_in_interval"].dt.hour


print("Exploded the hours into separate rows")


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
df_hours["date"] = pd.to_datetime(df_hours["date"])

election_days = ["2019-04-09", "2019-09-17", "2021-03-23", "2022-11-01", "2024-02-27"]
# Create the 'flag_weekend' column
df_hours["flag_weekend"] = (
    df_hours["date"].dt.weekday.isin([4, 5])
    | df_hours["date"].dt.date.isin(  # Check if Friday or Saturday
        israel_holidays.keys()
    )
    | df_hours["date"].dt.date.isin(  # Check if Friday or Saturday
        one_day_before_holidays # check if erev hag
    )
    | df_hours["date"].dt.date.isin(pd.to_datetime(election_days).date)  # 
)  # Check if elections day # Modified on 8/02/2025

df_hours["work_location_hours"] = df_hours.groupby(["date", "identifier"])[
    "flag_work_geohash"
].transform("sum")

df_hours["office_day"] = (df_hours["work_location_hours"] > 0) & (
    ~df_hours[
        "flag_weekend"
    ]  # corrected on 8th of february. but work location shouldnt be found on weekends
)

# Remove duplicates caused by consequent dwells, one of which starts and another one ends at the same hour
df_hours = df_hours.drop_duplicates(["identifier", "hour", "date"])

# calculate probability to be at home or at office in the specific hour
df_prob_full = conditional_probability_table(df_hours)
print(df_prob_full.head(2))
df_hours = df_hours.merge(
    df_prob_full[
        [
            "identifier",
            "month",
            "hour",
            "flag_weekend",
            # "expected_prob_office_hour",
            "prob_home_at_hour",
            "prob_office_at_hour",
            "prob_3rdplace_at_hour",
        ]
    ],
    on=["identifier", "month", "hour", "flag_weekend"],
    how="left",
).fillna(0)

df_monthly_remote, df_hours = process_month_remote_work(df_hours)

# df_hours.to_csv(utils.get_path("processed", "dwells", f"samples/df_hours_{MONTH}.csv"))
# print()

remote_work_stats_hour_date = (
    df_hours.groupby(["date", "hour", "identifier_type"])[
        ["flag_remote_work", "flag_work_hours", "flag_remote_work_smoothed"]
    ]
    .sum()
    .join(
        df_hours.groupby(["date", "hour", "identifier_type"])
        .size()
        .rename("count_dwells")
    )
    .reset_index()
)

df_monthly_remote.to_csv(
    utils.get_path("processed", "dwells", f"remote_work/user_remote_work_{MONTH}.csv")
)

remote_work_stats_hour_date.to_csv(
    utils.get_path(
        "processed", "dwells", f"remote_work/remote_work_stats_hour_date_{MONTH}.csv"
    )
)
print("files saved")
# Visualisation

df_month_loc_hours_office = (
    df_hours[
        (df_hours.month == MONTH) & (df_hours["office_day"]) & (~df_hours.flag_weekend)
    ]
    .groupby(["hour", "identifier_type", "location_type"])
    .size()
    .reset_index()
)
df_month_loc_hours = (
    df_hours[(df_hours.month == MONTH) & (~df_hours.flag_weekend)]
    .groupby(["hour", "identifier_type", "location_type"])
    .size()
    .reset_index()
)
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
sns.barplot(
    data=df_month_loc_hours_office[df_month_loc_hours_office.identifier_type == "GAID"],
    x="hour",
    y=0,
    hue="location_type",
    ax=axs[0, 0],
)
axs[0, 0].set_title(f"{MONTH} office days, Android users ")
sns.barplot(
    data=df_month_loc_hours_office[df_month_loc_hours_office.identifier_type == "IDFA"],
    x="hour",
    y=0,
    hue="location_type",
    ax=axs[1, 0],
)
axs[1, 0].set_title(f"{MONTH} office days, Iphone users ")
sns.barplot(
    data=df_month_loc_hours[df_month_loc_hours.identifier_type == "GAID"],
    x="hour",
    y=0,
    hue="location_type",
    ax=axs[0, 1],
)
axs[0, 1].set_title(f"{MONTH} workdays, Android users")
sns.barplot(
    data=df_month_loc_hours[df_month_loc_hours.identifier_type == "IDFA"],
    x="hour",
    y=0,
    hue="location_type",
    ax=axs[1, 1],
)
axs[1, 1].set_title(f"{MONTH} workdays, Iphone users")

plt.savefig(
    utils.get_path("processed", "dwells", f"plots/work_home_hours_{MONTH}.png"),
    format="png",
    dpi=300,
)
print("png saved")
