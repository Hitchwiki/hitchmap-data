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
from map_utils import *
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
    RationalQuadratic,
)
import pickle
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np
import seaborn as sns
import matplotlib
import shapely
from matplotlib.colors import LogNorm
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor

DAY = 24 * 60
WAIT_MAX = DAY


def get_points(path, wait_max=WAIT_MAX):
    points = gpd.read_file(path)
    points.wait = points.wait.astype(float)
    points.lat = points.lat.astype(float)
    points.lon = points.lon.astype(float)
    # threshold - assuming that values above that are skewed due to angriness of the hiker
    points = points[points["wait"] <= wait_max]

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
    points = get_points('../data/points_train_val.csv')
    points, polygon, map_boundary = get_points_in_region(points, region)
    points['lon'] = points.geometry.x
    points['lat'] = points.geometry.y

    train = get_points('../data/points_train.csv')
    train, polygon, map_boundary = get_points_in_region(train, region)
    train['lon'] = train.geometry.x
    train['lat'] = train.geometry.y
    
    val = get_points('../data/points_val.csv')
    val, polygon, map_boundary = get_points_in_region(val, region)
    val['lon'] = val.geometry.x
    val['lat'] = val.geometry.y

    return points, train, val

# centers data to a zero mean
class TargetTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, function=(lambda y: y), inverse_function=(lambda y: y)):
        self.function = function
        self.inverse_function = inverse_function
        self.mean = 0

    def fit(self, y):
         self.targets = y
        self.mean = np.mean(self.function(y))

    def transform(self, y):
        return self.function(y) - self.mean
        

    def inverse_transform(self, y):
        return self.inverse_function(y + self.mean)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

def evaluate(model, train, validation, features):
    train["pred"] = model.predict(train[features].values)

    print(
        f"Training RMSE: {root_mean_squared_error(train['wait'], train['pred'])}\n",
        f"Training MAE {mean_absolute_error(train['wait'], train['pred'])}",
    )

    validation["pred"] = model.predict(validation[features].values)

    print(
        f"Validation RMSE: {root_mean_squared_error(validation['wait'], validation['pred'])}\n",
        f"Validation MAE {mean_absolute_error(validation['wait'], validation['pred'])}",
    )

def evaluate_cv(estimator, X, y):
    cv_result = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=5,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=True
    )

    print(
        f"Training RMSE: {cv_result['train_neg_root_mean_squared_error'].mean() * -1}\n",
        f"Training MAE: {cv_result['train_neg_mean_absolute_error'].mean() * -1}\n",
        f"Cross-validation RMSE: {cv_result['test_neg_root_mean_squared_error'].mean() * -1}\n",
        f"Cross-validation MAE: {cv_result['test_neg_mean_absolute_error'].mean() * -1}",
    )


def get_optimized_gpr(initial_kernel, X, y):
    gpr = get_gpr(initial_kernel=initial_kernel)
    gpr.fit(X, y)

    return gpr


def get_gpr(initial_kernel):
    gpr = GaussianProcessRegressor(
        kernel=initial_kernel,
        alpha=0.0**2,
        optimizer="fmin_l_bfgs_b",
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=42,
    )

    log_transformer = TargetTransformer(function=np.log1p, inverse_function=np.expm1)
    target_transform_gpr = TransformedTargetRegressor(regressor=gpr, transformer=log_transformer)

    return target_transform_gpr