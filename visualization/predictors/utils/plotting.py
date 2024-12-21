import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.plot
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from numeric_transformers import exp_minus_tiny, log_plus_tiny
from utils_data import get_points
from utils_map import *
from utils_models import TargetTransformer


def plot_distribution_of_data_points():
    points = get_points("../data/points_train.csv")

    countries = gpd.read_file("map_features/countries/ne_110m_admin_0_countries.shp")
    countries = countries.to_crs(epsg=3857)
    countries = countries[countries.NAME != "Antarctica"]
    germany = countries[countries.NAME == "Germany"]
    europe_without_germany = countries[
        (countries.CONTINENT == "Europe") & (countries.NAME != "Germany")
    ]
    europe_without_germany_shape = europe_without_germany.geometry.unary_union

    world = countries[countries.CONTINENT != "Europe"]
    germany_data = points[points.geometry.within(germany.geometry.values[0])]
    europe_without_germany_data = points[
        points.geometry.within(europe_without_germany_shape)
    ]

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
    print(
        f"Europe without Germany: {round(len(europe_without_germany_data) / len(points) * 100, 2)} %"
    )
    print(f"Rest of the world: {round(len(world_data) / len(points) * 100, 2)} %")


def plot_1d_model_comparison(
    points, val, X, y, wag_model, average_model, tiles_model, gpr_model
):
    x_test = np.linspace(
        start=tiles_model.lon_intervals[0], stop=tiles_model.lon_intervals[-1], num=300
    )
    cut_through_germany = points.lat.values[0]
    x_test_2d_model = np.array([[xi, cut_through_germany] for xi in x_test])
    x_test = np.array([[xi] for xi in x_test])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.scatter(X, y, label="Observations")
    ax1.scatter(val.lon, val.wait, label="Validation")
    wag_pred = wag_model.predict(x_test_2d_model)
    ax1.plot(x_test, wag_pred, label="Weighted averaged Gaussian", color="red")
    ax1.set_ylim([0, 80])
    ax1.legend()
    ax1.set_xlabel("Longitude in m at 51° latitude")
    ax1.set_ylabel("Predicted waiting time")

    ax2.scatter(X, y, label="Observations")
    ax2.scatter(val.lon, val.wait, label="Validation")
    average_pred = average_model.predict(x_test_2d_model)
    ax2.plot(x_test, average_pred, label="Average", color="red")
    ax2.set_ylim([0, 80])
    ax2.legend()
    ax2.set_xlabel("Longitude in m at 51° latitude")
    ax2.set_ylabel("Predicted waiting time")

    ax3.scatter(X, y, label="Observations")
    ax3.scatter(val.lon, val.wait, label="Validation")
    tiles_pred = tiles_model.predict(x_test_2d_model)
    ax3.plot(x_test, tiles_pred, label="Tiles", color="red")
    ax3.set_ylim([0, 80])
    ax3.legend()
    ax3.set_xlabel("Longitude in m at 51° latitude")
    ax3.set_ylabel("Predicted waiting time")

    ax4.scatter(X, y, label="Observations")
    ax4.scatter(val.lon, val.wait, label="Validation")
    gpr_pred, _ = gpr_model.predict(x_test, return_std=True)
    ax4.plot(x_test, gpr_pred, label="GP mean prediction", color="red")
    ax4.set_ylim([0, 80])
    ax4.legend()
    ax4.set_xlabel("Longitude in m at 51° latitude")
    ax4.set_ylabel("Predicted waiting time")

    plt.show()


def plot_1d_with_uncertainties(gpr, X, y, start, stop):
    x_test = np.linspace(start=start, stop=stop, num=300)
    x_test = np.array([[xi] for xi in x_test])

    y_pred, std_prediction = gpr.predict(x_test, return_std=True, transform_predictions=False)

    #plt.scatter(X, y, label="Observations")
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


def plot_transformed_targets(X, y, start=0.3e6, stop=1.7e6):
    x_test = np.linspace(start=start, stop=stop, num=300)
    log_transformer = TargetTransformer(
        function=log_plus_tiny, inverse_function=exp_minus_tiny
    )
    y_transformed = log_transformer.fit_transform(y)

    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(X, y, label="Original data")
    ax1.plot(x_test, np.ones(x_test.shape[0]) * np.mean(y), label="Mean", color="red")
    ax1.legend()
    ax1.set_xlabel("Longitude in m at 51° latitude")
    ax1.set_ylabel("Waiting time")

    ax2.scatter(X, y_transformed, label="Transformed data", color="orange")
    ax2.plot(
        x_test,
        np.ones(x_test.shape[0]) * np.mean(y_transformed),
        label="Mean",
        color="red",
    )
    ax2.legend()
    ax2.set_xlabel("Longitude in m at 51° latitude")
    ax2.set_ylabel("Waiting time")

    plt.show()


def plot_lml_depending_on_lengthscale_noise(gpr):
    # from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-py
    lengthscale_space = np.logspace(2, 7, num=50)
    noise_level_space = np.logspace(-1, 1, num=50)
    lengthscale_grid, noise_level_grid = np.meshgrid(
        lengthscale_space, noise_level_space
    )

    log_marginal_likelihood = [
        gpr.regressor_.log_marginal_likelihood(theta=np.log([lengthscale, noise]))
        for lengthscale, noise in zip(
            lengthscale_grid.ravel(), noise_level_grid.ravel()
        )
    ]
    log_marginal_likelihood = np.reshape(
        log_marginal_likelihood, newshape=noise_level_grid.shape
    )

    vmin, vmax = (-log_marginal_likelihood).min(), (-log_marginal_likelihood).max()
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=50), decimals=1)
    plt.contour(
        lengthscale_grid,
        noise_level_grid,
        -log_marginal_likelihood,
        levels=level,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise-level")
    plt.title("Log-marginal-likelihood")
    plt.show()

def plot_rbf_covariance():
    def rbf(x, l, s=1.0):
        return s * np.exp(-0.5 * (x**2 / l**2))

    X_distance = np.logspace(0, 6, 100)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 5))

    lengthscale = 1.68e5
    ax1.plot(X_distance, rbf(X_distance, lengthscale, 1.0))
    ax1.set_xscale("log")
    ax1.axvline(x=lengthscale, color='red', label="Lengthscale")
    ax1.legend()
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("RBF covariance")
    

    lengthscale = 3.3e3
    ax2.plot(X_distance, rbf(X_distance, lengthscale, 1.0))
    ax2.set_xscale("log")
    ax2.axvline(x=lengthscale, color='red', label="Lengthscale")
    ax2.legend()
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("RBF covariance")

    plt.show()