import pandas as pd
import pygeohash
import pyproj
from shapely.geometry import Point, Polygon
import geopandas as gpd
from shapely.geometry import base

# from shapely.geometry import shape, GeometryCollection
from shapely import wkt, wkb
import os

import configparser

# from matplotlib import pyplot as plt


def get_path(data_type, source, filename, ini_file="settings.ini"):
    """
    Reads a path from the settings.ini file and optionally appends a subfolder.

    Args:
        section (str): The section in the ini file (e.g., 'Paths').
        data_type (str): type of data: processed/raw.
        source (str): A subfolder to append to the base path: census/gtfs/nadlan etc.
        ini_file (str): The path to the ini configuration file.

    Returns:
        str: The full path to the desired directory.
    """
    config = configparser.ConfigParser()
    config.read(ini_file)
    section = "Paths"
    if not config.has_section(section) or not config.has_option(section, data_type):
        raise ValueError(
            f"Section '{section}' or key '{data_type}' not found in {ini_file}."
        )

    # Get the base path
    base_path = os.path.abspath(config.get(section, data_type))

    # Append the subfolder if specified
    if source:
        base_path = os.path.join(base_path, f"data_{source}")

    return os.path.join(base_path, filename)


def make_gdf(df, geometry, crs="4326"):
    """_summary_

    Args:
        df (_type_): _description_
        geometry (_type_): _description_
        crs (str, optional): _description_. Defaults to "epsg:4326".

    Returns:
        _type_: _description_
    """
    if isinstance(df[geometry].iloc[0], base.BaseGeometry):
        return gpd.GeoDataFrame(df, geometry=geometry, crs=f"{crs}")
    elif isinstance(df[geometry].iloc[0], str):  # Likely WKT
        try:
            df[geometry] = df[geometry].apply(
                lambda x: None if pd.isnull(x) else wkt.loads(x)
            )
            print("Parsed WKT to geometry")
        except Exception as e:
            print(f"Failed to parse WKT: {e}")
    elif isinstance(df[geometry].iloc[0], (bytes, bytearray)):  # Likely WKB
        try:
            df[geometry] = df[geometry].apply(
                lambda x: None if pd.isnull(x) else wkb.loads(x)
            )
            print("Parsed WKB to geometry")
        except Exception as e:
            print(f"Failed to parse WKB: {e}")
    else:
        print(f"Unrecognized geometry type: {type(df[geometry].iloc[0])}")
        return df

    return gpd.GeoDataFrame(df, geometry=geometry, crs=f"{crs}")


def get_points(lat, long):
    """
    As input it accepts 2 Series
    """
    return [Point(j, i) for i, j in zip(lat, long)]


def get_point_32637(loc):
    #     print(loc.x, loc.y)
    inProj = pyproj.CRS("EPSG:4326")  # source coordinate system
    outProj = pyproj.CRS("EPSG:32637")  # destination coordinate system

    return Point(pyproj.transform(inProj, outProj, loc.y, loc.x))


def geohash_to_polygon(geohash):
    # Decode the geohash to get the bounding box
    lon, lat, lon_err, lat_err = pygeohash.decode_exactly(geohash)

    # Calculate the bounding box corners
    lat_min = lat - lat_err
    lat_max = lat + lat_err
    lon_min = lon - lon_err
    lon_max = lon + lon_err

    # Define the coordinates for the polygon using the bounding box corners
    coordinates = [
        (lon_min, lat_min),  # Bottom-left
        (lon_min, lat_max),  # Top-left
        (lon_max, lat_max),  # Top-right
        (lon_max, lat_min),  # Bottom-right
        (lon_min, lat_min),  # Close the polygon
    ]

    # Create a polygon from these coordinates
    polygon = Polygon(coordinates)
    return polygon


def dist_2fields(g):
    """calculate distance without changing crs.
    As input it accepts Series, usually row of 2 values
    """
    try:
        return g.iloc[0].distance(g.iloc[1])
    except ValueError:
        return 0


def area_2fields(g):
    """calculate area without changing crs.
    As input it accepts Series, usually row of 2 values
    """
    try:
        return g.iloc[0].intersection(g.iloc[1]).area
    except ValueError:
        return 0


# def get_h3_net(city_boarders):
#     """Create a h3 net for  given city borders.
#     borders type : [[]]
#     """
#     pass
#     boarders_gjson = {"type": "Polygon", "coordinates": [city_boarders]}
#     hexs = h3.polyfill(boarders_gjson, res=9, geo_json_conformant=True)

#     polygonise = lambda hex_id: Polygon(h3.h3_to_geo_boundary(hex_id, geo_json=True))

#     # %time all_polys = gpd.GeoSeries(list(map(polygonise, hexs)), \
#     #                                       index=hexs, \
#     #                                       crs="EPSG:4326" \
#     #                                      )

#     gdf_hex = gpd.GeoDataFrame(
#         list(map(polygonise, hexs)), index=hexs, crs="EPSG:4326", columns=["geometry"]
#     )
#     #     fig, ax=plt.subplots(figsize=(13,10))
#     #     gdf_hex.plot(alpha=0.5, linewidth=1, ax=ax)
#     #     ctx.add_basemap(ax=ax, crs='epsg:4326')
#     #     plt.show()

#     return gdf_hex
