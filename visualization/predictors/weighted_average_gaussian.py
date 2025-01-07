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

tqdm.pandas()

RESOLUTION = 2

# 180 degree meridian in epsg 3857
MERIDIAN = 20037508

class WeightedAveragedGaussian(MapBasedModel):
    def __init__(
        self, region="world", method="ordinary", resolution=RESOLUTION, verbose=False
    ):
        self.region = region
        self.method = method
        self.resolution = resolution  # pixel per degree
        self.verbose = verbose

    # https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    def makeGaussian(self, stdv, x0, y0):
        """Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        # TODO why fwhm used here?
        fwhm = 2.355 * stdv

        # gives the distribution in the whole raster space as X and Y are used here
        # TODO only calculate for pixels that will be colored (landmass) in the end
        # TODO only calculate for pixels that are significantly close to the point (e.g. 500 km around the point)
        return np.exp(
            -4 * np.log(2) * ((self.X - x0) ** 2 + (self.Y - y0) ** 2) / fwhm**2
        )

    # choose larger radius for not dense points (set radius till k other spots are in the radius)
    # scaling: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    def calc_radius(self, point, other_points, k=5):
        k = k + 1  # +1 as the point itself is also in the array
        distances = point.distance(other_points).to_numpy()
        idx = np.argpartition(distances, k)
        closest_distances = distances[idx[:k]]
        radius = np.ceil(max(closest_distances)) / FACTOR_STDV
        return radius

    def get_distribution(self, lon, lat):
        # standard deviation in meters -> 50 km around each spot; quite arbitrary choice
        if self.method == "DYNAMIC":
            stdv_m = max(
                self.circle_size,
                calc_radius(Point(lon, lat), self.points.geometry, k=K),
            )
        else:
            stdv_m = self.circle_size + 10000 * self.iteration
        return self.makeGaussian(stdv_m, lon, lat)

    # make_raster_map
    def fit(
        self,
        X,
        y,
        iteration=0,
        recompute=True,
        no_data_threshold=0.00000001,
    ):
        self.rasterio_raster = None
        self.raw_raster: np.array = None

        self.points = X
        self.circle_size = 50000
        self.iteration = iteration
        self.get_map_grid()

        # sum of distributions
        Zn = None
        # weighted sum of distributions
        Zn_weighted = None

        try:
            if recompute:
                raise Exception("recompute")
            else:
                Z = np.loadtxt(f"intermediate/map_{region}.txt", dtype=float)

        except:
            # create a raster map - resulution is defined above
            # https://stackoverflow.com/questions/56677267/tqdm-extract-time-passed-time-remaining
            if self.verbose:
                print("Weighting gaussians for all points...")
            with tqdm(
                zip(X[:, 0], X[:, 1], y), total=X.shape[0], disable=not self.verbose
            ) as t:
                # TODO find out how to speed up and parallelize this
                for lon, lat, wait in t:
                    # distribution inserted by a single point
                    Zi = self.get_distribution(lon, lat)
                    # add the new distribution to the sum of existing distributions
                    # write them to Zn_weighted and wait every single point/ distribution by the waiting time
                    # => it matters where a distribiton is inserted (areas with more distributions have a higher certainty)
                    # and which waiting time weight is associated with it
                    if Zn is None:
                        Zn = Zi
                        Zn_weighted = Zi * wait
                    else:
                        Zn = np.sum([Zn, Zi], axis=0)
                        Zn_weighted = np.sum([Zn_weighted, Zi * wait], axis=0)

                elapsed = t.format_dict["elapsed"]
                elapsed_str = t.format_interval(elapsed)
                df = pd.DataFrame(
                    {"region": self.region, "elapsed time": [elapsed_str]}
                )

                tracker_name = "logging/time_tracker.csv"
                try:
                    full_df = pd.read_csv(tracker_name, index_col=0)
                    full_df = pd.concat([full_df, df])
                    full_df.to_csv(tracker_name, sep=",")
                except:
                    df.to_csv(tracker_name)

            # normalize the weighted sum by the sum of all distributions -> so we see the actual waiting times in the raster
            Z = np.divide(
                Zn_weighted, Zn, out=np.zeros_like(Zn_weighted), where=Zn != 0
            )

            # grey out pixels with no hitchhiking spots near them
            undefined = -1.0
            Z = np.where(Zn < no_data_threshold, undefined, Z)

            # save the underlying raster data of the heatmap for later use
            np.savetxt(f"intermediate/map_{self.region}.txt", Z)

        self.raw_raster = Z
        self.save_as_rasterio()

        return self

    def predict(self, X):
        # from https://rasterio.readthedocs.io/en/stable/quickstart.html#spatial-indexing
        # in epsg 3857

        predictions = []

        for lon, lat in X:
            # transform the lat/lon to the raster's coordinate system
            x, y = self.rasterio_raster.index(lon, lat)
            # read the raster at the given coordinates
            predictions.append(self.rasterio_raster.read(1)[x, y])

        return np.array(predictions)