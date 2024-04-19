import rasterio
import rasterio.plot

from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import shapely

from shapely.geometry import Point

from utils_map import *


DAY = 24 * 60
WAIT_MAX = DAY


def get_points(path, wait_max=WAIT_MAX):
    points = gpd.read_file(path)
    points.wait = points.wait.astype(float)
    points.lat = points.lat.astype(float)
    points.lon = points.lon.astype(float)
    # threshold - assuming that values above that are skewed due to angriness of the hiker
    points = points[points["wait"] <= wait_max]
    points = points[points["lat"] < 70]  # removing the point on Greenland

    # use epsg 3857 as default as it gives coordinates in meters
    points.geometry = gpd.points_from_xy(points.lon, points.lat)
    points.crs = CRS.from_epsg(4326)
    points = points.to_crs(epsg=3857)

    return points


def get_cut_through_germany():
    region = "germany"

    points = get_points("../data/points_train.csv")
    points, polygon, map_boundary = get_points_in_region(points, region)
    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y

    val = get_points("../data/points_val.csv")
    val, polygon, map_boundary = get_points_in_region(val, region)
    val["lon"] = val.geometry.x
    val["lat"] = val.geometry.y

    vertical_cut = 6621293  # cutting Germany vertically through Dresden
    offset = 10000  # 10km strip

    points = points[
        (points.lat > vertical_cut - offset) & (points.lat < vertical_cut + offset)
    ]
    points.geometry = points.geometry.map(
        lambda point: shapely.ops.transform(lambda x, y: (x, vertical_cut), point)
    )
    points["lat"] = vertical_cut
    val = val[(val.lat > vertical_cut - offset) & (val.lat < vertical_cut + offset)]
    val.geometry = val.geometry.map(
        lambda point: shapely.ops.transform(lambda x, y: (x, vertical_cut), point)
    )
    val["lat"] = vertical_cut

    # ->
    test_start = 0.4e6
    test_stop = 1.7e6

    return points, val


def get_from_region(region):
    points = get_points("../data/points_train_val.csv")
    points, polygon, map_boundary = get_points_in_region(points, region)
    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y

    train = get_points("../data/points_train.csv")
    train, polygon, map_boundary = get_points_in_region(train, region)
    train["lon"] = train.geometry.x
    train["lat"] = train.geometry.y

    val = get_points("../data/points_val.csv")
    val, polygon, map_boundary = get_points_in_region(val, region)
    val["lon"] = val.geometry.x
    val["lat"] = val.geometry.y

    return points, train, val