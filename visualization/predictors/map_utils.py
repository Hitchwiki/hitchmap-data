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
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from models import *

from tqdm.auto import tqdm

tqdm.pandas()

RESOLUTION = 2


def save_numpy_map(
    map, region="world", method="ordinary", kind_of_map="map", resolution=RESOLUTION
):
    map_path = f"intermediate/{kind_of_map}_{method}_{region}_{resolution}.txt"
    np.savetxt(map_path, map)


def load_numpy_map(
    region="world", method="ordinary", kind_of_map="map", resolution=RESOLUTION
):
    map_path = f"intermediate/{kind_of_map}_{method}_{region}_{resolution}.txt"
    return np.loadtxt(map_path)


def load_raster(region="world", method="ordinary", resolution=RESOLUTION):
    map_path = f"intermediate/map_{method}_{region}_{resolution}.tif"
    return rasterio.open(map_path)


def get_points_in_region(points, region="world"):
    # set lat long boundaries of different scopes of the map

    maps = {
        "germany": [3.0, 48.0, 16.0, 55.0],
        "spain": [-8.0, 36.0, 3.0, 43.0],
        "spain_france": [-8.0, 36.0, 6.0, 50.0],
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
    # nodes stars bottom left & clockwise
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


def make_map_from_gp(gp, average, region, polygon, map_boundary, resolution=10):
    X, Y = get_map_grid(polygon, map_boundary, resolution)
    grid = np.array((Y, X)).T
    map = np.empty((0, X.shape[0]))
    certainty_map = np.empty((0, X.shape[0]))

    for vertical_line in tqdm(grid):
        pred, stdv = gp.predict(vertical_line, return_std=True)
        pred = np.exp(pred + average)
        map = np.vstack((map, pred))
        certainty_map = np.vstack((certainty_map, stdv))

    map = map.T
    certainty_map = certainty_map.T

    save_numpy_map(map, region=region, method="gp", resolution=resolution)
    save_numpy_map(
        certainty_map,
        region=region,
        method="gp",
        kind_of_map="certainty",
        resolution=resolution,
    )
    map = save_as_raster(
        map, polygon, map_boundary, region=region, method="gp", resolution=resolution
    )
    plt.contourf(X, Y, map)
    plt.colorbar()
    plt.savefig(f"maps/contourf_map_gp_{region}_{resolution}.png")


def raster_from_model(model, region, resolution=RESOLUTION, show_uncertainties=False):
    model_name = type(model).__name__

    raster_maker = MapBasedModel(model_name, region, resolution)

    raster_maker.get_map_grid()
    X, Y = raster_maker.X, raster_maker.Y
    grid = np.array((X, Y))
    height = X.shape[0]
    map = np.empty((0, height))
    if show_uncertainties:
        uncertainty_map = np.empty((0, height))

    # transposing the grid enables us to iterate over it vertically
    # and single elements become lon-lat pairs that can be fed into the model
    for vertical_line in tqdm(grid.transpose(), disable=not raster_maker.verbose):
        if show_uncertainties:
            pred, stdv = model.predict(vertical_line, return_std=True)
            uncertainty_map = np.vstack((uncertainty_map, stdv))
        else:
            pred = model.predict(vertical_line)
        map = np.vstack((map, pred))

    # because we vstacked above
    map = map.T
    if show_uncertainties:
        uncertainty_map = uncertainty_map.T

    save_numpy_map(map, region=region, method=model_name, resolution=resolution)
    if show_uncertainties:
        save_numpy_map(
            uncertainty_map,
            region=region,
            method=model_name,
            kind_of_map="uncertainty",
            resolution=resolution,
        )

    raster_maker.raw_raster = map
    raster_maker.raw_uncertaities = uncertainty_map

    raster_maker.save_as_raster()

    return raster_maker


def map_from_model(model, region, resolution=RESOLUTION, show_uncertainties=False):
    raster_maker = raster_from_model(
        model, region, resolution, show_uncertainties=show_uncertainties
    )
    raster_maker.build_map(show_uncertainties=show_uncertainties)
