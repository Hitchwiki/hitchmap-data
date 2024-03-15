import sys
import numpy
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

DAY = 24 * 60
WAIT_MAX = DAY

def get_points(path, wait_max=WAIT_MAX):
    points = gpd.read_file(path)
    points.wait = points.wait.astype(float)
    points.lat = points.lat.astype(float)
    points.lon = points.lon.astype(float)
    # threshold - assuming that values above that are skewed due to angriness of the hiker
    points = points[points['wait'] <= wait_max]

    # use epsg 3857 as default as it gives coordinates in meters
    points.geometry = gpd.points_from_xy(points.lon, points.lat)
    points.crs = CRS.from_epsg(4326)
    points = points.to_crs(epsg=3857)

    return points

class GP():
    def __init__():
        pass

    def fit(X, y):
        C = get_kernelmatrix(X)
    
    def rbf_kernel(x1, y2, l):
        return l * exp((-1) * distance(x1, x2)**2)

    def get_kernelmatrix(X):
        pass

    def pred():
        return mean, variance
