import rasterio
import rasterio.plot
from rasterio.crs import CRS
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import rasterio.mask
from shapely.geometry import Point
from geopandas import GeoDataFrame
import matplotlib.colors as colors
from matplotlib import cm
from shapely.geometry import Polygon
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
import osmnx
from pathlib import Path


RESOLUTION = 10  # pixel per degree


def define_raster(polygon, map, res=RESOLUTION):
    xx, yy = polygon.geometry[0].exterior.coords.xy

    # Note above return values are of type `array.array`
    xx = xx.tolist()
    yy = yy.tolist()

    degree_width = int(map[2] - map[0])
    degree_height = int(map[3] - map[1])
    pixel_width = degree_width * res
    pixel_height = degree_height * res

    return xx, yy, pixel_width, pixel_height


def save_raster(Z, polygon, map):

    polygon_vertices_x, polygon_vertices_y, pixel_width, pixel_height = define_raster(
        polygon, map
    )
    # https://gis.stackexchange.com/questions/425903/getting-rasterio-transform-affine-from-lat-and-long-array

    # lower/upper - left/right
    ll = (polygon_vertices_x[0], polygon_vertices_y[0])
    ul = (polygon_vertices_x[1], polygon_vertices_y[1])  # in lon, lat / x, y order
    ur = (polygon_vertices_x[2], polygon_vertices_y[2])
    lr = (polygon_vertices_x[3], polygon_vertices_y[3])
    cols, rows = pixel_width, pixel_height

    # ground control points
    gcps = [
        GCP(0, 0, *ul),
        GCP(0, cols, *ur),
        GCP(rows, 0, *ll),
        GCP(rows, cols, *lr),
    ]

    # seems to need the vertices of the map polygon
    transform = from_gcps(gcps)

    # cannot use np.longdouble to write to tif
    Z = np.double(Z)
    Z = np.round(Z, 0)

    # save the colored raster using the above transform
    # TODO find out why raster is getting smaller in x direction when stored as tif (e.g. 393x700 -> 425x700)
    with rasterio.open(
        map_path,
        "w",
        driver="GTiff",
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        crs=CRS.from_epsg(3857),
        transform=transform,
        dtype=Z.dtype,
    ) as destination:
        destination.write(Z, 1)


def get_map_grid(polygon, map, res=RESOLUTION):
    xx, yy, pixel_width, pixel_height = define_raster(polygon, map, res)
    x = np.linspace(xx[0], xx[2], pixel_width)
    # mind starting with upper value of y axis here
    y = np.linspace(yy[2], yy[0], pixel_height)
    X, Y = np.meshgrid(x, y)
    # higher precision prevents pixels far away from the points to be 0/ nan
    X = np.longdouble(X)
    Y = np.longdouble(Y)
    return X, Y


def get_points_in_region(points, region='world'):
    # set lat long boundaries of different scopes of the map

    maps = {
        "germany": [5.0, 48.0, 15.0, 55.0],
        "europe": [-12.0, 35.0, 45.0, 71.0],
        "world": [-180.0, -85.0, 180.0, 85.0],  # 85 lat bc of 3857
        "small": [12.0, 52.0, 15.0, 54.0],
        "africa": [-20.0, -35.0, 60.0, 40.0],
        "asia": [40.0, 0.0, 180.0, 85.0],
        "north_america": [-180.0, 0.0, -20.0, 85.0],
        "south_america": [-90.0, -60.0, -30.0, 15.0],
        "australia": [100.0, -50.0, 180.0, 0.0],
        "middle_africa": [-10.0, -35.0, 60.0, 20.0],
        "artificial": [8.0, -1.0, 30.0, 1.0],
        "greenland": [-80.0, 60.0, -10.0, 85.0],
    }
    map_boundary = maps[region]

    # create boundary polygon
    polygon = Polygon(
        [
            (map_boundary[0], map_boundary[1]),
            (map_boundary[0], map_boundary[3]),
            (map_boundary[2], map_boundary[3]),
            (map_boundary[2], map_boundary[1]),
            (map_boundary[0], map_boundary[1]),
        ]
    )
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])
    polygon = polygon.to_crs(epsg=3857)

    # extract points within polygon
    points = points[points.geometry.within(polygon.geometry[0])]

    return points, polygon, map_boundary
