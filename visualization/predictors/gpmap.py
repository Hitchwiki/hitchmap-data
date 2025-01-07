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
from map_based_model import MapBasedModel
from utils.utils_models import fit_gpr_silent
import glob

class GPMap(MapBasedModel):
    def __init__(self, region="world", resolution=10, version="prod"):
        self.gpr_path = "models/kernel.pkl"
        self.points_path = "dump.sqlite"

        with open(self.gpr_path, "rb") as file:
            self.gpr = pickle.load(file)

        super().__init__(method=type(self.gpr).__name__, region=region, resolution=resolution, version=version, verbose=False)
        
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
                if self.recalc_raster[y][x] == 0:
                    continue
                this_point = [float(coords[0]), float(coords[1])]
                to_predict.append(this_point)
                pixels_to_predict.append((y, x))
                # batching the model calls
                if len(to_predict) == self.batch_size:
                    prediction = self.gpr.predict(np.array(to_predict), return_std=False)
                    for i, (y, x) in enumerate(pixels_to_predict):
                        self.raw_raster[y][x] = prediction[i]

                    to_predict = []
                    pixels_to_predict = []
        
        if len(to_predict) > 0:
            prediction = self.gpr.predict(np.array(to_predict), return_std=False)
            for i, (y, x) in enumerate(pixels_to_predict):
                self.raw_raster[y][x] = prediction[i]

        print(f"Time elapsed to compute full map: {time.time() - start}")
        print(
            f"For map of shape: {self.raw_raster.shape} that is {self.raw_raster.shape[0] * self.raw_raster.shape[1]} pixels and an effective time per pixel of {(time.time() - start) / (self.raw_raster.shape[0] * self.raw_raster.shape[1])} seconds"
        )
        print((f"Only {self.recalc_raster.sum()} pixels were recalculated. That is {self.recalc_raster.sum() / (self.raw_raster.shape[0] * self.raw_raster.shape[1]) * 100}% of the map."))
        print(f"And time per recalculated pixel was {(time.time() - start) / self.recalc_raster.sum()} seconds")

        np.savetxt(self.map_path, self.raw_raster)
        self.save_as_rasterio()

    def show_raster(self, raster):
        """Show the raster in a plot."""
        plt.imshow(raster, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def pixel_from_point(self, point) -> tuple[int, int]:
        """For a given point by coordinates, determines the pixel in the raster that best corresponds to it."""
        lats = self.Y.transpose()[0]
        lat_index = None
        for i, lat in enumerate(lats):
            if lat >= point["lat"] and point["lat"] >= lats[i+1]:
                lat_index = i
                break

        lons = self.X[0]
        lon_index = None
        for i, lon in enumerate(lons):
            if lon <= point["lon"] and point["lon"] <= lons[i+1]:
                lon_index = i
                break

        result = (lat_index, lon_index)

        return result
        
    def get_recalc_raster(self):
        """Creats 2d np.array of raster where only pixels that are 1 should be recalculated."""
        
        recalc_radius_pixels = int(np.ceil(abs(self.recalc_radius / (self.grid[0][0][0] - self.grid[0][0][1]))))

        self.recalc_raster = np.zeros(self.grid.shape[1:])

        new_points = get_points(self.points_path, begin=self.begin, until=self.today)
        new_points["lon"] = new_points.geometry.x
        new_points["lat"] = new_points.geometry.y
        print(f"Recalculating map for {len(new_points)} new points.")
        for i, point in new_points.iterrows():
            lat_pixel, lon_pixel = self.pixel_from_point(point)

            for i in range(lat_pixel - recalc_radius_pixels, lat_pixel + recalc_radius_pixels):
                for j in range(lon_pixel - recalc_radius_pixels, lon_pixel + recalc_radius_pixels):
                    if i < 0 or j < 0 or i >= self.recalc_raster.shape[0] or j >= self.recalc_raster.shape[1]:
                        continue
                    self.recalc_raster[i, j] = 1
        self.show_raster(self.recalc_raster)
        
        print("Report reduction of rasters.")
        print(self.recalc_raster.sum(), self.recalc_raster.shape[0] * self.recalc_raster.shape[1], self.recalc_raster.sum() / (self.recalc_raster.shape[0] * self.recalc_raster.shape[1]))
        self.get_landmass_raster()
        self.recalc_raster = self.recalc_raster * self.landmass_raster
        self.show_raster(self.recalc_raster)
        print(self.landmass_raster.sum(), self.landmass_raster.shape[0] * self.landmass_raster.shape[1], self.landmass_raster.sum() / (self.landmass_raster.shape[0] * self.landmass_raster.shape[1]))
        print(self.recalc_raster.sum(), self.recalc_raster.shape[0] * self.recalc_raster.shape[1], self.recalc_raster.sum() / (self.recalc_raster.shape[0] * self.recalc_raster.shape[1]))

    def get_landmass_raster(self):
        """Creates raster of landmass as np.array"""
        self.landmass_raster = np.ones(self.grid.shape[1:])

        polygon_vertices_x, polygon_vertices_y, pixel_width, pixel_height = (
            self.define_raster()
        )

        # handling special case when map spans over the 180 degree meridian
        if polygon_vertices_x[0] > 0 and polygon_vertices_x[2] < 0:
            polygon_vertices_x[2] = 2 * MERIDIAN + polygon_vertices_x[2]
            polygon_vertices_x[3] = 2 * MERIDIAN + polygon_vertices_x[3]

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

        # cannot use np.float128 to write to tif
        self.landmass_raster = self.landmass_raster.astype(np.float64)

        # save the colored raster using the above transform
        # important: rasterio requires [0,0] of the raster to be in the upper left corner and [rows, cols] in the lower right corner
        # TODO find out why raster is getting smaller in x direction when stored as tif (e.g. 393x700 -> 425x700)
        with rasterio.open(
            self.landmass_path,
            "w",
            driver="GTiff",
            height=self.landmass_raster.shape[0],
            width=self.landmass_raster.shape[1],
            count=1,
            crs=CRS.from_epsg(3857),
            transform=transform,
            dtype=self.landmass_raster.dtype,
        ) as destination:
            destination.write(self.landmass_raster, 1)

        landmass_rasterio = rasterio.open(self.landmass_path)

        nodata = 0

        countries = gpd.read_file(
            "map_features/countries/ne_110m_admin_0_countries.shp"
        )
        countries = countries.to_crs(epsg=3857)
        countries = countries[countries.NAME != "Antarctica"]
        country_shapes = countries.geometry
        country_shapes = country_shapes.apply(lambda x: make_valid(x))

        out_image, out_transform = rasterio.mask.mask(
            landmass_rasterio, country_shapes, nodata=nodata
        )

        self.landmass_raster = out_image[0]
        self.show_raster(self.landmass_raster)

        # cleanup
        os.remove(self.landmass_path)


def recalc():
    gpmap = GPMap(resolution=1)
    gpmap.recalc_map()