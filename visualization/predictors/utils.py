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
    points = points[points['wait'] <= wait_max]
    points = points[points['lat'] < 70] # removing the point on Greenland

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


def evaluate(model, train, validation, features=['lon', 'lat']):
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


def evaluate_cv(estimator, X, y, folds=5):
    cv_result = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=folds,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=True,
        return_estimator=True,
    )

    print(
        "Cross-validated averaged metrics...\n",
        f"Training RMSE: {cv_result['train_neg_root_mean_squared_error'].mean() * -1}\n",
        f"Training MAE: {cv_result['train_neg_mean_absolute_error'].mean() * -1}\n",
        f"Validation RMSE: {cv_result['test_neg_root_mean_squared_error'].mean() * -1}\n",
        f"Validation MAE: {cv_result['test_neg_mean_absolute_error'].mean() * -1}\n",
    )

    # returning one of the estimators for visualization purposes
    return cv_result["estimator"][0]


def get_log_transformer():
    return TargetTransformer(function=np.log1p, inverse_function=np.expm1)


def get_gpr(initial_kernel):
    gpr = GaussianProcessRegressor(
        kernel=initial_kernel,
        alpha=0.0**2,
        optimizer="fmin_l_bfgs_b",
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=42,
    )

    log_transformer = get_log_transformer()
    target_transform_gpr = TransformedTargetRegressor(
        regressor=gpr, transformer=log_transformer
    )

    return target_transform_gpr


def get_optimized_gpr(initial_kernel, X, y):
    gpr = get_gpr(initial_kernel=initial_kernel)
    gpr.fit(X, y)

    return gpr


def plot_distribution_of_data_points():
    points = get_points("../data/points_train.csv")

    countries = gpd.datasets.get_path("naturalearth_lowres")
    countries = gpd.read_file(countries)
    countries = countries.to_crs(epsg=3857)
    countries = countries[countries.name != "Antarctica"]
    germany = countries[countries.name == "Germany"]
    europe_without_germany = countries[(countries.continent == "Europe") & (countries.name != "Germany")]
    europe_without_germany_shape = europe_without_germany.geometry.unary_union

    world = countries[countries.continent != "Europe"]
    germany_data = points[points.geometry.within(germany.geometry.values[0])]
    europe_without_germany_data = points[points.geometry.within(europe_without_germany_shape)]

    europe = pd.concat([germany, europe_without_germany])
    europe_data = pd.concat([germany_data, europe_without_germany_data])

    world_data = points[~(points.index.isin(europe_data.index))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    europe.plot(ax=ax1, facecolor="none", edgecolor="black")
    germany_data.plot(ax=ax1, markersize=0.01, color="red")
    europe_without_germany_data.plot(ax=ax1, markersize=0.01, color="blue")
    ax1.set_xlim([-0.3e7, 0.6e7])
    ax1.set_ylim([0.4e7, 1.2e7])

    countries.plot(ax=ax2, facecolor="none", edgecolor="black")
    world_data.plot(ax=ax2, markersize=0.01, color="green")
    europe_data.plot(ax=ax2, markersize=0.01, color="blue")

    plt.show()

    print(f"Germany: {round(len(germany_data) / len(points) * 100, 2)} %")
    print(f"Europe without Germany: {round(len(europe_without_germany_data) / len(points) * 100, 2)} %")
    print(f"Rest of the world: {round(len(world_data) / len(points) * 100, 2)} %")

def plot_1d_model_comparison(points, val, X, y, wag_model, average_model, tiles_model, gpr_model):
    x_test = np.linspace(start=tiles_model.lon_intervals[0], stop=tiles_model.lon_intervals[-1], num=300)
    cut_through_germany = points.lat.values[0]
    x_test_2d_model = np.array([[xi, cut_through_germany] for xi in x_test])
    x_test = np.array([[xi] for xi in x_test])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.scatter(X, y, label="Observations")
    ax1.scatter(val.lon, val.wait, label="Validation")
    wag_pred = wag_model.predict(x_test_2d_model)
    ax1.plot(x_test, wag_pred, label="Weighted averaged Gaussian", color='red')
    ax1.set_ylim([0, 80])
    ax1.legend()
    ax1.set_xlabel("Longitude in m at 51° latitude")
    ax1.set_ylabel("Predicted waiting time")

    ax2.scatter(X, y, label="Observations")
    ax2.scatter(val.lon, val.wait, label="Validation")
    average_pred = average_model.predict(x_test_2d_model)
    ax2.plot(x_test, average_pred, label="Average", color='red')
    ax2.set_ylim([0, 80])
    ax2.legend()
    ax2.set_xlabel("Longitude in m at 51° latitude")
    ax2.set_ylabel("Predicted waiting time")

    ax3.scatter(X, y, label="Observations")
    ax3.scatter(val.lon, val.wait, label="Validation")
    tiles_pred = tiles_model.predict(x_test_2d_model)
    ax3.plot(x_test, tiles_pred, label="Tiles", color='red')
    ax3.set_ylim([0, 80])
    ax3.legend()
    ax3.set_xlabel("Longitude in m at 51° latitude")
    ax3.set_ylabel("Predicted waiting time")

    ax4.scatter(X, y, label="Observations")
    ax4.scatter(val.lon, val.wait, label="Validation")
    gpr_pred = gpr_model.predict(x_test)
    ax4.plot(x_test, gpr_pred, label="GP mean prediction", color='red')
    ax4.set_ylim([0, 80])
    ax4.legend()
    ax4.set_xlabel("Longitude in m at 51° latitude")
    ax4.set_ylabel("Predicted waiting time")

    plt.show()

def plot_1d_with_uncertainties(gpr, X, y, start, stop):
    x_test = np.linspace(start=start, stop=stop, num=300)
    x_test = np.array([[xi] for xi in x_test])

    y_pred, std_prediction = gpr.predict(x_test, return_std=True)

    plt.scatter(X, y, label="Observations")
    plt.plot(x_test, y_pred, label="Mean prediction")
    plt.fill_between(
        x_test.ravel(),
        y_pred - 1.96 * std_prediction,
        y_pred + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )

    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.show()