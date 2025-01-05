import geopandas as gpd
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.plot
from matplotlib import cm
from matplotlib import pyplot as plt
from rasterio.control import GroundControlPoint as GCP
from rasterio.crs import CRS
from rasterio.transform import from_gcps
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm.auto import tqdm
import time
import os
import pickle
from utils.utils_data import get_points
from models import MapBasedModel
from utils.utils_models import fit_gpr_silent
import glob

class GPMap(MapBasedModel):
    def __init__(self):
        self.gpr_path = "models/kernel.pkl"
        self.points_path = "dump.sqlite"

        with open(self.gpr_path, "rb") as file:
            self.gpr = pickle.load(file)

        super().__init__(method=type(self.gpr).__name__, region="world", resolution=10, version="prod", verbose=False)
        
        files = glob.glob(f"intermediate/map_{self.method}_{self.region}_{self.resolution}_{self.version}*.txt")
        if len(files) == 0:
            raise FileNotFoundError("No base map calculated so far.")
        else:
            latest_date = pd.Timestamp.min
            for file in files:
                date = pd.Timestamp(file.split("_")[-1].split(".")[0])
                if date > latest_date:
                    latest_date = date
                    self.old_map_path = file

        self.begin = latest_date

        self.batch_size = 10000
        self.today = pd.Timestamp("2024-3-30")
        self.map_path = f"intermediate/map_{self.method}_{self.region}_{self.resolution}_{self.version}_{self.today.date()}.txt"

        self.recalc_radius = 800000 # TODO: determine from model largest influence radius

    def recalc_map(self):
        """Recalculate the map with the current Gaussian Process model.

        Overrides the stored np.array raster of the map.
        """
        # fit model to new data points

        self.points = get_points(self.points_path, until=self.today)
        self.points["lon"] = self.points.geometry.x
        self.points["lat"] = self.points.geometry.y

        X = self.points[["lon", "lat"]].values
        y = self.points["wait"].values  

        self.gpr.regressor.optimizer = None
        self.gpr = fit_gpr_silent(self.gpr, X, y)

        # recalc the old map

        self.raw_raster = np.loadtxt(self.old_map_path)

        self.get_map_grid()
        self.get_recalc_raster()

        print("Compute pixels that are expected to differ...")
        start = time.time()
        to_predict = []
        pixels_to_predict = []
        for x, vertical_line in tqdm(
            enumerate(self.grid.transpose()), total=len(self.grid.transpose())
        ):
            for y, coords in enumerate(vertical_line):
                if self.recalc_landmass[y][x] == 0:
                    continue
                this_point = [float(coords[0]), float(coords[1])]
                to_predict.append(this_point)
                pixels_to_predict.append((y, x))
                # batching the model calls
                if len(to_predict) == self.batch_size:
                    prediction = model.predict(np.array(to_predict), return_std=False)
                    for i, (y, x) in enumerate(pixels_to_predict):
                        self.raw_raster[y][x] = prediction[i]

                    to_predict = []
                    pixels_to_predict = []
                
        prediction = model.predict(np.array(to_predict), return_std=False)
        for i, (y, x) in enumerate(pixels_to_predict):
            self.raw_raster[y][x] = prediction[i]

        print(f"Time elapsed to compute full map: {time.time() - start}")
        print(
            f"For map of shape: {self.raw_raster.shape} that is {self.raw_raster.shape[0] * self.raw_raster.shape[1]} pixels and an effective time per pixel of {(time.time() - start) / (self.raw_raster.shape[0] * self.raw_raster.shape[1])} seconds"
        )
        print((f"Only {recalc_landmass.sum()} pixels were recalculated. That is {recalc_landmass.sum() / (self.raw_raster.shape[0] * self.raw_raster.shape[1]) * 100}% of the map."))
        print(f"And time per recalculated pixel was {(time.time() - start) / recalc_landmass.sum()} seconds")

        np.savetxt(self.map_path, self.raw_raster)
        self.save_as_raster()


    def get_recalc_raster(self):
        """Creats 2d np.array of raster where only pixels that are 1 should be recalculated."""
        def pixel_from_point(point) -> tuple[int, int]:
            lats = map.Y.transpose()[0]
            lat_index = None
            for i, lat in enumerate(lats):
                if lat >= point["lat"] and point["lat"] >= lats[i+1]:
                    lat_index = i
                    break

            lons = map.X[0]
            lon_index = None
            for i, lon in enumerate(lons):
                if lon <= point["lon"] and point["lon"] <= lons[i+1]:
                    lon_index = i
                    break

            return (lat_index, lon_index)
        
        recalc_radius_pixels = int(np.ceil(abs(self.recalc_radius / (self.grid[0][0][0] - self.grid[0][0][1]))))

        self.recalc_raster = np.zeros(self.grid.shape[1:])
        self.recalc_raster.shape
        new_points = get_points(self.points_path, begin=self.begin, until=self.today)
        for i, point in new_points.iterrows():
            lat_pixel, lon_pixel = pixel_from_point(point)

            for i in range(lat_pixel - recalc_radius_pixels, lat_pixel + recalc_radius_pixels):
                for j in range(lon_pixel - recalc_radius_pixels, lon_pixel + recalc_radius_pixels):
                    if i < 0 or j < 0 or i >= self.recalc_raster.shape[0] or j >= self.recalc_raster.shape[1]:
                        continue
                    self.recalc_raster[i, j] = 1
        
        print("Report reduction of rasters.")
        print(self.recalc_raster.sum(), self.recalc_raster.shape[0] * self.recalc_raster.shape[1], self.recalc_raster.sum() / (self.recalc_raster.shape[0] * self.recalc_raster.shape[1]))
        self.get_landmass_raster()
        self.recalc_raster = self.recalc_raster * self.landmass_raster
        print(self.landmass_raster.sum(), self.landmass_raster.shape[0] * self.landmass_raster.shape[1], self.landmass_raster.sum() / (self.landmass_raster.shape[0] * self.landmass_raster.shape[1]))
        print(self.recalc_raster.sum(), self.recalc_raster.shape[0] * self.recalc_raster.shape[1], self.recalc_raster.sum() / (self.recalc_raster.shape[0] * self.recalc_raster.shape[1]))
