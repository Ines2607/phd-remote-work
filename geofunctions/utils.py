import pandas as pd
import pygeohash
import pyproj
from shapely.geometry import Point, Polygon, MultiPolygon
import geopandas as gpd
from shapely.geometry import base

# from shapely.geometry import shape, GeometryCollection
from shapely import wkt, wkb
import os

import configparser
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# from matplotlib import pyplot as plt


def get_path(data_type, source, filename="", ini_file="settings.ini"):
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
    config.read(f"../{ini_file}")
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
        df (pd.DataFrame): _description_
        geometry (str): _description_
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
    """_summary_

    Args:
        geohash (str): _description_

    Returns:
        polygon: geometry in crs 4326
    """
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


def spatial_variance(group):
    """

    Args:
        group (_type_): _description_

    Returns:
        _variance: spatial variance from home in km
    """
    distances = group["geometry"].distance(group["geometry_home"])
    # Compute variance of distances
    variance = distances.var() / (10**6)
    return variance


def calculate_distance(gdf, col1, col2, crs=4326):
    """
    Calculate the distance between geometries in two columns of a GeoDataFrame,
    ensuring each column is reprojected separately.

    Parameters:
        gdf (gpd.GeoDataFrame or pd.DataFrame): The DataFrame containing the geometry columns.
        col1 (str): Name of the first geometry column.
        col2 (str): Name of the second geometry column.

    Returns:
        gpd.GeoSeries: A Series of distances.
    """
    # Ensure the GeoDataFrame has a CRS set
    if isinstance(gdf, pd.DataFrame) or gdf.crs is None:
        gdf = make_gdf(gdf, geometry=col1, crs=crs)
    # Handle the first geometry column
    geo1 = gpd.GeoSeries(gdf[col1], crs=gdf.crs)
    if not geo1.crs or not geo1.crs.is_projected:
        geo1 = geo1.to_crs(gdf.estimate_utm_crs())

    # Handle the second geometry column
    geo2 = gpd.GeoSeries(gdf[col2], crs=gdf.crs)
    if not geo2.crs or not geo2.crs.is_projected:
        geo2 = geo2.to_crs(gdf.estimate_utm_crs())

    # Calculate distances between geometries
    distances = geo1.distance(geo2)

    return distances


def clip_gdf_gushdan_boundaries(
    gdf,
    geometry="geometry",
):
    """
    Clips a GeoDataFrame to the boundaries of Gush Dan.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to clip.
        geometry (str): The name of the geometry column in the GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: The clipped GeoDataFrame.
    """
    # Ensure geometry column exists
    if geometry not in gdf.columns:
        raise ValueError(f"Column '{geometry}' not found in GeoDataFrame.")

    # Read Gush Dan boundaries
    try:
        gdf_boundaries = gpd.read_file(
            get_path("processed", "adm", "gushdan_polygon.geojson")
        )
    except Exception as e:
        raise FileNotFoundError("Unable to read Gush Dan boundaries file.") from e

    if gdf_boundaries.empty:
        raise ValueError("The Gush Dan boundaries file is empty.")

    # Extract and buffer the boundaries geometry
    buffer_distance = 0.001
    gdf_boundaries[geometry] = gdf_boundaries.geometry.buffer(buffer_distance)

    # Ensure CRS compatibility
    if gdf.crs != gdf_boundaries.crs:
        print(f"Converting CRS from {gdf.crs} to {gdf_boundaries.crs}")
        gdf = gdf.to_crs(gdf_boundaries.crs)

    # Clip the GeoDataFrame to the boundaries
    gdf_clipped = clip_gdf(gdf, gdf_boundaries)

    return gdf_clipped


def clip_gdf(gdf, boundaries, geometry="geometry", how="inner"):
    """
    Clips a GeoDataFrame using the centroids of its geometries if they are Polygons or directly
    uses the geometries if they are Points. If boundaries is a GeoDataFrame, performs a spatial join.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to clip.
        boundaries (Polygon, GeoDataFrame): The boundary geometry to clip against.
            - If a Polygon, then transform to GeoDataFrame with CRS 4326 .
            - If a GeoDataFrame, it performs a spatial join to clip.
        geometry (str): The name of the geometry column in the GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: The clipped GeoDataFrame with original geometries.
    """
    # Ensure geometry column exists
    if geometry not in gdf.columns:
        raise ValueError(f"Column '{geometry}' not found in GeoDataFrame.")

    if isinstance(boundaries, (Polygon, MultiPolygon)):
        boundaries = gpd.GeoDataFrame(boundaries, crs="EPSG:4326")

    # Handle boundaries as GeoDataFrame
    if isinstance(boundaries, gpd.GeoDataFrame):
        # # Remove all geometry-like columns except the one used in gdf.geometry
        drop_columns = [
            col
            for col in boundaries.select_dtypes("geometry").columns
            if col != boundaries.geometry.name
        ]
        if len(drop_columns) > 0:
            boundaries = boundaries.drop(drop_columns, axis=1)

        # Ensure CRS matches
        if gdf.crs != boundaries.crs:
            print(f"Converting CRS from {gdf.crs} to {boundaries.crs}")
            gdf = gdf.to_crs(boundaries.crs)

        gdf["centroid"] = gdf[geometry].centroid  # Create centroids
        # Perform spatial join
        if len(boundaries) == 0:
            raise ValueError("The boundaries GeoDataFrame is empty.")

        gdf_within_boundaries = gpd.sjoin(
            gdf.set_geometry("centroid"), boundaries, how=how, predicate="within"
        )
        print("joined")
        gdf_within_boundaries = gdf_within_boundaries.drop(
            columns=["centroid"]
        ).set_geometry(geometry)

        print(
            f"Original shape: {gdf.shape[0]}, new shape: {gdf_within_boundaries.shape[0]}"
        )
    else:
        raise ValueError("Boundaries must be a Polygon or GeoDataFrame.")

    return gdf_within_boundaries


def return_list_months_ok():
    return [
        "201909",
        "201911",
        "201912",
        "202001",
        "202002",
        "202003",
        "202004",
        "202005",
        "202006",
        "202007",
        "202008",
        "202009",
        "202010",
        "202011",
        "202101",
        "202102",
        "202103",
        "202104",
        "202105",
        "202106",
        "202107",
        "202108",
        "202109",
        "202110",
        "202111",
        "202112",
        "202201",
        "202202",
        "202203",
        "202204",
        "202205",
        "202206",
        "202207",
        "202208",
        "202209",
        "202210",
        "202211",
        "202301",
        "202302",
        "202303",
        "202304",
        "202305",
        "202306",
        "202309",
    ]


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
