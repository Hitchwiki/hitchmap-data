from utils import *
from map_utils import *
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
    ExpSineSquared,
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

# log space
points = get_points("../data/points_train.csv")
# for log transformation make sure waiting times are above 0 otherwise we get -inf for those values
points["wait"] = points["wait"].apply(lambda x: 0.1 if x == 0 else x)
region = "europe"
points, polygon, map_boundary = get_points_in_region(points, region)
points["lat"] = points.geometry.y
points["lon"] = points.geometry.x

X = points[["lat", "lon"]].values
y = points["wait"].values

# y_ being the waiting time in the (new) log space
# log space allows us to contrain the prior of functions (in the original space)
# to functions being always positive
# underlying cause: the original ys are not normally distributed (as they do not become less than 0)
y_ = np.log(y)

# assuming mean = 0 in the gp setup
average = np.mean(y_)
y_ = y_ - average
stdv = np.std(y_)

# parameters to optimize

l = 1e5
L = [l, l]
sigma = stdv

rqk = RationalQuadratic(
    length_scale=1e5,
    alpha=1.0,
    length_scale_bounds=(1e0, 1e6),
    alpha_bounds=(1e-05, 100000.0),
)

kernel = ConstantKernel(
    constant_value=sigma**2, constant_value_bounds=(1e-1, 1e1)
) * rqk + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-2, 1e2))

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.0**2,
    optimizer="fmin_l_bfgs_b",  # maximizing marginal log lik: given the data, how likely is the parameterization of the kernel
    # (if we draw a sample frome the multivariate gaussian with the kernel as covariance matrix, how likely is the data we have seen)
    # prevents overfitting to some degree
    normalize_y=False,
    n_restarts_optimizer=0,
    random_state=42,
)

gp.fit(X, y_)
print(gp.kernel_, np.exp(gp.kernel_.theta))





# training error
points["pred"], points["std"] = gp.predict(X, return_std=True)
points["pred"] = np.exp(points["pred"] + average)

print(
    (
        mean_squared_error(points["wait"], points["pred"]),
        root_mean_squared_error(points["wait"], points["pred"]),
        mean_absolute_error(points["wait"], points["pred"]),
    )
)

# validation error
val = get_points("../data/points_val.csv")
val, polygon, map_boundary = get_points_in_region(val, region)
val["lat"] = val.geometry.y
val["lon"] = val.geometry.x

val["pred"], val["std"] = gp.predict(val[["lat", "lon"]].values, return_std=True)
val["pred"] = np.exp(val["pred"] + average)

print(
    (
        mean_squared_error(val["wait"], val["pred"]),
        root_mean_squared_error(val["wait"], val["pred"]),
        mean_absolute_error(val["wait"], val["pred"]),
    )
)

m = make_map_from_gp(gp, average, region, polygon, map_boundary, resolution=2)
