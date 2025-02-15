import os

import argparse
import dask.dataframe as dd
import dask_geopandas as dg
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import argparse
import datetime
import pygeohash
from shapely import wkb
from geofunctions import utils
import ipdb

## only inside gushdan
def filter_dwells_by_coords(df):
    # Create geometry from latitude and longitude
    df["the_geom"] = df.map_partitions(
        lambda part: gpd.GeoSeries(
            [Point(xy) for xy in zip(part.centroid_longitude, part.centroid_latitude)]
        ),
        meta=gpd.GeoSeries(),
    )

    # Convert the DataFrame to a Dask GeoDataFrame if necessary
    df = dg.from_dask_dataframe(df)

    # Set geometry column
    df = df.set_geometry("the_geom")

    # Filter by spatial boundary
    df = df[df.within(boundaries)]

    return df


# we have to had at least 4 signals ( 2 home, 2 work)
def filter_dwells_by_id(df, min_count=4):
    # Group by 'identifier' and calculate the size
    counts = df.groupby("identifier").size().reset_index(name="frequency")

    # Merge the count data back into the original DataFrame
    df = df.merge(counts, on="identifier")

    # Filter based on the frequency count
    return df[df["frequency"] >= min_count]


def add_time_flags(df, start_col="start_date_time", end_col="end_date_time"):

    # Extract hour and weekday from start and end times
    start_hour = df[start_col].dt.hour
    start_date = df[start_col].dt.date
    end_hour = df[end_col].apply(lambda x: x.hour if pd.notnull(x) else None)
    end_date = df[end_col].apply(lambda x: x.date() if pd.notnull(x) else None)
    start_day = df[start_col].dt.dayofweek

    # Define night hours
    night_start_hour = 22  # 10 PM
    night_end_hour = 7  # 7 AM

    # Define work hours and days
    work_start_hour = 9  # 9 AM
    work_end_hour = 19  # 7 PM

    # Flag for night hours (10 PM - 7 AM)
    df["flag_night"] = (
        (start_hour >= night_start_hour)
        | (start_hour < night_end_hour)
        | (end_hour >= night_start_hour)
        | (end_hour < night_end_hour)
        | (
            (start_hour < night_start_hour)
            & (end_hour > night_end_hour)
            & (start_date != end_date)
        )
    )

    # Flag for work hours (Sunday to Thursday, 9 AM - 7 PM)
    df["flag_work_hours"] = (
        ((start_day < 4) | (start_day == 6))
        & (start_hour < work_end_hour)
        & (end_hour >= work_start_hour)
    )
    return df


# PATH

target_folder = utils.get_path("raw", "dwells")

geometry_folder = utils.get_path("processed", "adm")
source_folder = target_folder

# Boundaries of Gush Dan
boundaries = gpd.read_file(os.path.join(geometry_folder, "gushdan_polygon.geojson"))
boundaries = boundaries.iloc[0]["geometry"]


# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Read a Parquet file with a specified month filter."
)
parser.add_argument(
    "--quarter", type=int, required=True, help="Specify the month to filter by (1-12)"
)


# Parse arguments
args = parser.parse_args()
QUARTER = args.quarter

quarterfolder = f"{QUARTER}_full.parquet"
file_path = os.path.join(source_folder, quarterfolder)
list_month_numbers = [
    int(name.split("month=")[1]) for name in os.listdir(file_path) if "month" in name
]

for MONTH_NUMBER in list_month_numbers:

    YEAR = int(str(QUARTER)[:4])
    if MONTH_NUMBER == 12:
        YEAR -= 1

    if MONTH_NUMBER < 10:
        MONTH = f"{YEAR}0{MONTH_NUMBER}"
    else:
        MONTH = f"{YEAR}{MONTH_NUMBER}"

    filteredfile = os.path.join(source_folder, f"data_filtered_{MONTH}.parquet")

    # Check if the file exists
    print(filteredfile)
    if not os.path.exists(filteredfile):

        df = dd.read_parquet(file_path, filters=[("month", "==", MONTH_NUMBER)])
        print(f"we read file for {MONTH} ")
        print(f"original size {df.shape}")
        # Basic filters

        df_filt = filter_dwells_by_coords(df)
        print(f"after filter by coords size {df_filt.shape}")

        df_filt = df_filt.compute()

        df_filt.columns

        df_filtered_full = filter_dwells_by_id(df_filt)

        print("we filtered file")

        with open(os.path.join(target_folder, "file_monthly_info.txt"), "a") as file:
            file.write(
                f"Month: {MONTH}, Original Size: {df.shape[0].compute()}, Original  Users: {df.identifier.nunique().compute()}, Filtered Size: {df_filtered_full.shape[0]}, Filtered Users: {df_filtered_full.identifier.nunique()}\n"
            )

        # Create features

        df_filtered_full["duration_hours"] = df_filtered_full["duration_seconds"] / (
            60 * 60
        )

        df_filtered_full = df_filtered_full[
            (df_filtered_full.classification.isin(["AREA_DWELL"])|(df_filtered_full.classification.isin(["DWELL"])))
        ]

        df_filtered_full = df_filtered_full.set_crs(4326)

        # Ideally, we need DBSCAN but the dataset is to big for it
        #  precision=7 around 150 meters
        df_filtered_full["geohash"] = [
            pygeohash.encode(lon, lat, precision=7)
            for lat, lon in zip(
                df_filtered_full["centroid_latitude"],
                df_filtered_full["centroid_longitude"],
            )
        ]
       
        df_filtered_full["date"] = pd.to_datetime(
            df_filtered_full["local_date_time"]
        ).dt.date  # start time

        df_filtered_full["start_date_time"] = pd.to_datetime(
            df_filtered_full["local_date_time"]
        )
        df_filtered_full["end_date_time"] = [
            start + datetime.timedelta(seconds=sec)
            for start, sec in zip(
                df_filtered_full["start_date_time"],
                df_filtered_full["duration_seconds"],
            )
        ]

        # Apply the function
        df_filtered_full = add_time_flags(df_filtered_full)
        print(df_filtered_full.shape)
        # ipdb.set_trace()

        # Create filters for home and work hours

        filter_string_home = (df_filtered_full.classification.isin(["AREA_DWELL"])|(df_filtered_full.classification.isin(["DWELL"]))) & (
            df_filtered_full["flag_night"]
        )
        filter_string_work = (df_filtered_full.classification.isin(["AREA_DWELL"])|(df_filtered_full.classification.isin(["DWELL"]))) & (
            df_filtered_full["flag_work_hours"]
        )

        # Create pivot table by user and geohash

        index_columns = ["identifier", "geohash"]

        df_night_stays = (
            df_filtered_full[filter_string_home]
            .groupby(index_columns)["date"]
            .nunique()
            .to_frame()
            .join(
                df_filtered_full[filter_string_home]
                .groupby(index_columns)["duration_seconds"]
                .sum()
                .to_frame()
            )
            .join(
                df_filtered_full[filter_string_home]
                .groupby(index_columns)["bump_count"]
                .sum()
                .to_frame()
            )
        )

        df_work_stays = (
            df_filtered_full[filter_string_work]
            .groupby(index_columns)["date"]
            .nunique()
            .to_frame()
            .join(
                df_filtered_full[filter_string_work]
                .groupby(index_columns)["duration_seconds"]
                .sum()
                .to_frame()
            )
            .join(
                df_filtered_full[filter_string_work]
                .groupby(index_columns)["bump_count"]
                .sum()
                .to_frame()
            )
        )

        # Final filters. We have to see user at least 2 times per months during home and work hours to conclude about home and work

        df_max_nights = df_night_stays.groupby(["identifier"])["date"].max()

        ids_night = df_max_nights[df_max_nights > 1].index

        df_max_works = df_work_stays.groupby(["identifier"])["date"].max()

        ids_work = df_max_works[df_max_works > 1].index

        good_ids = set(ids_work).intersection(set(ids_night))

        len(good_ids)

        print(f"{len(good_ids)} ids of original  ids")

        df_night_stays_fltrd = df_night_stays.loc[list(good_ids)]
        df_work_stays_fltrd = df_work_stays.loc[list(good_ids)]

        df_night_stays_fltrd.shape

        df_work_stays_fltrd.shape

        df_filtered_good = df_filtered_full[
            df_filtered_full.identifier.isin(list(good_ids))
        ]

        # df_filtered_good = df_filtered_good.set_crs(4326)

        df_filtered_good.shape[0] / df_filtered_full.shape[0]

        # keep all types of dwells for future analysis but only "good" ids
        df_filtered_good2save = df_filtered_good.copy()
        df_filtered_good2save["the_geom"] = df_filtered_good2save["the_geom"].apply(
            wkb.dumps
        )

        df_filtered_good2save.to_parquet(filteredfile)

        print("partquet saved")

    else:
        print("flitered file already exists")
